from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


LABEL_MAP = {
    "Best-RGB-optimized/ppo": "PPO-RGB",
    "Best-Greyscale-optimized/ppo": "PPO-GRY",
    "Best-Masked-optimized/ppo": "PPO-MSK",
    "Best-RGB-optimized/dqn": "DQN-RGB",
    "Best-Masked-optimized/dqn": "DQN-MSK",
    "Best-Greyscale-optimized/dqn": "DQN-GRY",
}


def agent_group(label: str) -> int:
    return 0 if label.startswith("PPO-") else 1

SETTING_ORDER = ["rgb", "grayscale", "masked"]
SECTION_TITLE = {
    "rgb": "RGB Environment",
    "grayscale": "Grayscale Environment",
    "masked": "Masked Environment",
}
OWN_VS_MASKED_TARGETS = [
    ("Best-RGB-optimized/ppo", "rgb"),
    ("Best-RGB-optimized/dqn", "rgb"),
    ("Best-Greyscale-optimized/ppo", "grayscale"),
    ("Best-Greyscale-optimized/dqn", "grayscale"),
]
MASKED_TRANSFER_TARGETS = [
    ("ppo", "rgb", "Best-RGB-optimized/ppo", "Best-Masked-optimized/ppo"),
    ("dqn", "rgb", "Best-RGB-optimized/dqn", "Best-Masked-optimized/dqn"),
    ("ppo", "grayscale", "Best-Greyscale-optimized/ppo", "Best-Masked-optimized/ppo"),
    ("dqn", "grayscale", "Best-Greyscale-optimized/dqn", "Best-Masked-optimized/dqn"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un report per ambiente usando i best trial.")
    parser.add_argument("--input-dir", default="models_checkpoints")
    parser.add_argument("--output-dir", default="results/best_trial_env_report")
    parser.add_argument("--title", default="Best Trial Cross-Environment Report")
    return parser.parse_args()


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def load_all_best_trial_eval(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    seed_rows: list[dict] = []

    for json_path in sorted(input_dir.glob("Best-*-optimized/*/best_trial/*_model_eval_*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        checkpoint_parts = Path(data["checkpoint"].replace("\\", "/")).parts
        model_folder = next(part for part in checkpoint_parts if part.startswith("Best-"))
        agent = str(data["agent"]).lower()
        short_label = f"{model_folder}/{agent}"
        display_label = LABEL_MAP.get(short_label, short_label)
        run_id = f"{short_label}/{data['setting']}"

        for seed_block in data["per_seed"]:
            seed = int(seed_block["seed"])
            rewards = [float(x) for x in seed_block["episode_rewards"]]
            seed_rows.append(
                {
                    "setting": data["setting"],
                    "run_id": run_id,
                    "short_label": short_label,
                    "display_label": display_label,
                    "seed": seed,
                    "mean_reward_seed": float(np.mean(rewards)),
                    "std_reward_seed": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
                    "episodes_in_seed": len(rewards),
                }
            )
            for episode_index, reward in enumerate(rewards, start=1):
                episode_rows.append(
                    {
                        "setting": data["setting"],
                        "run_id": run_id,
                        "short_label": short_label,
                        "display_label": display_label,
                        "seed": seed,
                        "episode_index": episode_index,
                        "reward": reward,
                    }
                )

    return pd.DataFrame(episode_rows), pd.DataFrame(seed_rows)


def add_ci(mean: float, sample_std: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return mean, mean
    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin = float(t_crit * sample_std / np.sqrt(n))
    return mean - margin, mean + margin


def build_environment_summary(seed_df: pd.DataFrame, episode_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_df.groupby(["setting", "run_id", "short_label", "display_label"], as_index=False)
        .agg(
            mean_seed_reward=("mean_reward_seed", "mean"),
            std_seed_reward=("mean_reward_seed", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    episode_summary = (
        episode_df.groupby(["setting", "run_id"], as_index=False)
        .agg(
            episode_mean_reward=("reward", "mean"),
            episode_std_reward=("reward", "std"),
            total_episodes=("reward", "size"),
        )
    )
    summary = summary.merge(episode_summary, on=["setting", "run_id"], how="left")
    ci_bounds = [
        add_ci(
            float(row.mean_seed_reward),
            float(0.0 if pd.isna(row.std_seed_reward) else row.std_seed_reward),
            int(row.n_seeds),
        )
        for row in summary.itertuples(index=False)
    ]
    summary["ci_low"] = [low for low, _ in ci_bounds]
    summary["ci_high"] = [high for _, high in ci_bounds]
    summary = summary.sort_values(["setting", "mean_seed_reward"], ascending=[True, False]).reset_index(drop=True)
    summary["rank_in_setting"] = summary.groupby("setting").cumcount() + 1
    return summary


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def build_environment_tests(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows: list[dict] = []
    pairwise_rows: list[dict] = []

    for setting in SETTING_ORDER:
        subset = seed_df[seed_df["setting"] == setting].copy()
        pivot = subset.pivot(index="seed", columns="display_label", values="mean_reward_seed").dropna()
        ordered_models = (
            subset.groupby("display_label", as_index=False)["mean_reward_seed"]
            .mean()
            .sort_values("mean_reward_seed", ascending=False)["display_label"]
            .tolist()
        )
        pivot = pivot[ordered_models]

        statistic, p_value = stats.friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
        omnibus_rows.append(
            {
                "setting": setting,
                "test": "friedmanchisquare",
                "n_models": pivot.shape[1],
                "n_seeds": pivot.shape[0],
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

        raw_p_values: list[float] = []
        local_rows: list[dict] = []
        for left_index, left_model in enumerate(ordered_models):
            left = pivot[left_model].to_numpy()
            for right_model in ordered_models[left_index + 1:]:
                right = pivot[right_model].to_numpy()
                diffs = left - right
                if np.allclose(diffs, 0.0):
                    w_stat, w_p = 0.0, 1.0
                else:
                    w_stat, w_p = stats.wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
                local_rows.append(
                    {
                        "setting": setting,
                        "model_a": left_model,
                        "model_b": right_model,
                        "mean_a": float(np.mean(left)),
                        "mean_b": float(np.mean(right)),
                        "mean_diff_a_minus_b": float(np.mean(diffs)),
                        "wilcoxon_statistic": float(w_stat),
                        "p_value_raw": float(w_p),
                    }
                )
                raw_p_values.append(float(w_p))

        adjusted = holm_adjust(raw_p_values)
        for row, p_adj in zip(local_rows, adjusted):
            row["p_value_holm"] = float(p_adj)
            row["significant_0_05"] = bool(p_adj < 0.05)
            pairwise_rows.append(row)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def build_outcome_matrix(summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, setting: str) -> pd.DataFrame:
    models = summary_df[summary_df["setting"] == setting].sort_values("rank_in_setting")["display_label"].tolist()
    models = sorted(models, key=lambda label: (agent_group(label), models.index(label)))
    matrix = pd.DataFrame("Pareggio", index=models, columns=models)
    for model in models:
        matrix.loc[model, model] = "-"

    subset = pairwise_df[pairwise_df["setting"] == setting]
    for row in subset.itertuples(index=False):
        if row.significant_0_05:
            if row.mean_diff_a_minus_b > 0:
                matrix.loc[row.model_a, row.model_b] = "Vittoria"
                matrix.loc[row.model_b, row.model_a] = "Sconfitta"
            elif row.mean_diff_a_minus_b < 0:
                matrix.loc[row.model_a, row.model_b] = "Sconfitta"
                matrix.loc[row.model_b, row.model_a] = "Vittoria"
            else:
                matrix.loc[row.model_a, row.model_b] = "Pareggio"
                matrix.loc[row.model_b, row.model_a] = "Pareggio"
        else:
            matrix.loc[row.model_a, row.model_b] = "Pareggio"
            matrix.loc[row.model_b, row.model_a] = "Pareggio"

    matrix.insert(0, "Model", matrix.index)
    return matrix.reset_index(drop=True)


def build_own_vs_masked_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    raw_p_values: list[float] = []

    for short_label, own_setting in OWN_VS_MASKED_TARGETS:
        display_label = LABEL_MAP.get(short_label, short_label)
        own_rows = seed_df[
            (seed_df["short_label"] == short_label) & (seed_df["setting"] == own_setting)
        ][["seed", "mean_reward_seed"]].rename(columns={"mean_reward_seed": "own_reward"})
        masked_rows = seed_df[
            (seed_df["short_label"] == short_label) & (seed_df["setting"] == "masked")
        ][["seed", "mean_reward_seed"]].rename(columns={"mean_reward_seed": "masked_reward"})

        paired = own_rows.merge(masked_rows, on="seed", how="inner").sort_values("seed")
        own_values = paired["own_reward"].to_numpy()
        masked_values = paired["masked_reward"].to_numpy()
        diffs = own_values - masked_values

        if len(diffs) == 0 or np.allclose(diffs, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(
                own_values,
                masked_values,
                zero_method="wilcox",
                alternative="two-sided",
            )

        rows.append(
            {
                "model": display_label,
                "family": "RGB" if own_setting == "rgb" else "Grayscale",
                "own_environment": own_setting,
                "masked_environment": "masked",
                "mean_reward_own": float(np.mean(own_values)),
                "std_reward_own": float(np.std(own_values, ddof=1)) if len(own_values) > 1 else 0.0,
                "mean_reward_masked": float(np.mean(masked_values)),
                "std_reward_masked": float(np.std(masked_values, ddof=1)) if len(masked_values) > 1 else 0.0,
                "mean_diff_own_minus_masked": float(np.mean(diffs)) if len(diffs) else 0.0,
                "wilcoxon_statistic": float(statistic),
                "p_value_raw": float(p_value),
                "n_seeds": int(len(paired)),
            }
        )
        raw_p_values.append(float(p_value))

    adjusted = holm_adjust(raw_p_values)
    for row, p_adj in zip(rows, adjusted):
        row["p_value_holm"] = float(p_adj)
        row["significant_0_05"] = bool(p_adj < 0.05)
        if not row["significant_0_05"]:
            row["outcome"] = "Pareggio"
        elif row["mean_diff_own_minus_masked"] > 0:
            row["outcome"] = "Vittoria own env"
        elif row["mean_diff_own_minus_masked"] < 0:
            row["outcome"] = "Vittoria masked"
        else:
            row["outcome"] = "Pareggio"

    return pd.DataFrame(rows)


def build_masked_transfer_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    raw_p_values: list[float] = []

    for agent, eval_setting, optimized_label, masked_label in MASKED_TRANSFER_TARGETS:
        optimized_display = LABEL_MAP.get(optimized_label, optimized_label)
        masked_display = LABEL_MAP.get(masked_label, masked_label)

        optimized_rows = seed_df[
            (seed_df["short_label"] == optimized_label) & (seed_df["setting"] == eval_setting)
        ][["seed", "mean_reward_seed"]].rename(columns={"mean_reward_seed": "optimized_reward"})
        masked_rows = seed_df[
            (seed_df["short_label"] == masked_label) & (seed_df["setting"] == eval_setting)
        ][["seed", "mean_reward_seed"]].rename(columns={"mean_reward_seed": "masked_reward"})

        paired = optimized_rows.merge(masked_rows, on="seed", how="inner").sort_values("seed")
        optimized_values = paired["optimized_reward"].to_numpy()
        masked_values = paired["masked_reward"].to_numpy()
        diffs = optimized_values - masked_values

        if len(diffs) == 0 or np.allclose(diffs, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(
                optimized_values,
                masked_values,
                zero_method="wilcox",
                alternative="two-sided",
            )

        rows.append(
            {
                "agent": agent.upper(),
                "environment": eval_setting,
                "optimized_model": optimized_display,
                "masked_model": masked_display,
                "mean_reward_optimized": float(np.mean(optimized_values)),
                "std_reward_optimized": float(np.std(optimized_values, ddof=1)) if len(optimized_values) > 1 else 0.0,
                "mean_reward_masked": float(np.mean(masked_values)),
                "std_reward_masked": float(np.std(masked_values, ddof=1)) if len(masked_values) > 1 else 0.0,
                "mean_diff_optimized_minus_masked": float(np.mean(diffs)) if len(diffs) else 0.0,
                "wilcoxon_statistic": float(statistic),
                "p_value_raw": float(p_value),
                "n_seeds": int(len(paired)),
            }
        )
        raw_p_values.append(float(p_value))

    adjusted = holm_adjust(raw_p_values)
    for row, p_adj in zip(rows, adjusted):
        row["p_value_holm"] = float(p_adj)
        row["significant_0_05"] = bool(p_adj < 0.05)
        if not row["significant_0_05"]:
            row["outcome"] = "Pareggio"
        elif row["mean_diff_optimized_minus_masked"] > 0:
            row["outcome"] = "Vittoria optimized"
        elif row["mean_diff_optimized_minus_masked"] < 0:
            row["outcome"] = "Vittoria masked"
        else:
            row["outcome"] = "Pareggio"

    return pd.DataFrame(rows)


def dataframe_to_tabularx(df: pd.DataFrame, font_size: str = r"\footnotesize", column_sep: int = 5) -> str:
    spec = ">{\\raggedright\\arraybackslash}p{0.22\\linewidth}" + ">{\\centering\\arraybackslash}p{0.11\\linewidth}" * (len(df.columns) - 1)
    header = " & ".join(rf"\textbf{{{latex_escape(col)}}}" for col in df.columns) + r" \\"
    rows = []
    for row in df.itertuples(index=False, name=None):
        values = [rf"{{\footnotesize {latex_escape(value)}}}" for value in row]
        rows.append(" & ".join(values) + r" \\")
    return "\n".join(
        [
            r"{\renewcommand{\arraystretch}{1.12}",
            font_size,
            r"\rowcolors{2}{black!3}{white}",
            rf"\setlength{{\tabcolsep}}{{{column_sep}pt}}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"}",
        ]
    )


def pairwise_matrix_to_tabular(df: pd.DataFrame, font_size: str = r"\scriptsize", column_sep: int = 3) -> str:
    columns = df.columns.tolist()
    row_labels = df["Model"].tolist()
    ppo_count = sum(label.startswith("PPO-") for label in row_labels)

    spec_parts = [">{\\raggedright\\arraybackslash}p{0.22\\linewidth}"]
    for idx in range(1, len(columns)):
        prefix = "|" if idx == ppo_count + 1 else ""
        spec_parts.append(f"{prefix}>{{\\centering\\arraybackslash}}p{{0.11\\linewidth}}")
    spec = "".join(spec_parts)

    header_parts = []
    for idx, col in enumerate(columns):
        cell = rf"\textbf{{{latex_escape(col)}}}"
        if idx == ppo_count + 1:
            cell = rf"\multicolumn{{1}}{{|c}}{{{cell}}}"
        header_parts.append(cell)
    header = " & ".join(header_parts) + r" \\"

    rows = []
    for row_idx, row in enumerate(df.itertuples(index=False, name=None), start=1):
        values = []
        for col_idx, value in enumerate(row):
            cell = rf"{{\footnotesize {latex_escape(value)}}}"
            if col_idx == ppo_count + 1:
                cell = rf"\multicolumn{{1}}{{|c}}{{{cell}}}"
            values.append(cell)
        rows.append(" & ".join(values) + r" \\")
        if row_idx == ppo_count:
            rows.append(r"\midrule")

    return "\n".join(
        [
            r"{\renewcommand{\arraystretch}{1.12}",
            font_size,
            r"\rowcolors{2}{black!3}{white}",
            rf"\setlength{{\tabcolsep}}{{{column_sep}pt}}",
            r"\resizebox{\linewidth}{!}{%",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"}",
        ]
    )


def build_section(summary_df: pd.DataFrame, omnibus_df: pd.DataFrame, pairwise_df: pd.DataFrame, setting: str) -> str:
    setting_summary = summary_df[summary_df["setting"] == setting].copy()
    stats_table = setting_summary[
        ["rank_in_setting", "display_label", "mean_seed_reward", "std_seed_reward", "ci_low", "ci_high", "n_seeds"]
    ].copy()
    stats_table = stats_table.rename(
        columns={
            "rank_in_setting": "Rank",
            "display_label": "Model",
            "mean_seed_reward": "Mean reward",
            "std_seed_reward": "Std dev",
            "ci_low": "CI low",
            "ci_high": "CI high",
            "n_seeds": "Seeds",
        }
    )
    for col in ["Mean reward", "Std dev", "CI low", "CI high"]:
        stats_table[col] = stats_table[col].map(lambda x: f"{float(x):.3f}")

    omnibus_row = omnibus_df[omnibus_df["setting"] == setting].iloc[0]
    omnibus_table = pd.DataFrame(
        [
            {
                "Test": "Friedman",
                "Statistic": f"{float(omnibus_row['statistic']):.4f}",
                "p-value": f"{float(omnibus_row['p_value']):.6f}",
                "Models": int(omnibus_row["n_models"]),
                "Seeds": int(omnibus_row["n_seeds"]),
            }
        ]
    )

    outcome_matrix = build_outcome_matrix(summary_df, pairwise_df, setting)

    return "\n".join(
        [
            rf"\section{{{SECTION_TITLE[setting]}}}",
            r"\subsection{Descriptive Statistics}",
            r"\begin{table}[H]",
            rf"\caption{{Descriptive statistics for the best trials in the {latex_escape(SECTION_TITLE[setting])}.}}",
            r"\centering",
            dataframe_to_tabularx(stats_table, font_size=r"\footnotesize", column_sep=5),
            r"\end{table}",
            r"\subsection{Omnibus Test}",
            r"\begin{table}[H]",
            rf"\caption{{Omnibus statistical test in the {latex_escape(SECTION_TITLE[setting])}.}}",
            r"\centering",
            dataframe_to_tabularx(omnibus_table, font_size=r"\footnotesize", column_sep=6),
            r"\end{table}",
            r"\subsection{Pairwise Outcomes}",
            r"\begin{table}[H]",
            rf"\caption{{Pairwise statistical outcomes in the {latex_escape(SECTION_TITLE[setting])}.}}",
            r"\centering",
            pairwise_matrix_to_tabular(outcome_matrix, font_size=r"\scriptsize", column_sep=3),
            r"\end{table}",
        ]
    )


def build_own_vs_masked_section(comparison_df: pd.DataFrame) -> str:
    display_df = comparison_df[
        [
            "model",
            "own_environment",
            "mean_reward_own",
            "std_reward_own",
            "mean_reward_masked",
            "std_reward_masked",
            "mean_diff_own_minus_masked",
            "p_value_holm",
            "outcome",
        ]
    ].copy()
    display_df = display_df.rename(
        columns={
            "model": "Model",
            "own_environment": "Own env",
            "mean_reward_own": "Mean own",
            "std_reward_own": "Std own",
            "mean_reward_masked": "Mean masked",
            "std_reward_masked": "Std masked",
            "mean_diff_own_minus_masked": "Diff own-masked",
            "p_value_holm": "Holm p-value",
            "outcome": "Outcome",
        }
    )
    for col in ["Mean own", "Std own", "Mean masked", "Std masked", "Diff own-masked", "Holm p-value"]:
        display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}")

    rgb_df = display_df[display_df["Own env"] == "rgb"].reset_index(drop=True)
    grayscale_df = display_df[display_df["Own env"] == "grayscale"].reset_index(drop=True)

    return "\n".join(
        [
            r"\section{Own Environment vs Masked}",
            r"This section compares the RGB and Grayscale best-trial models between their native evaluation environment and the masked environment, using paired seed-level rewards.",
            r"\subsection{RGB Models}",
            r"\begin{table}[H]",
            r"\caption{RGB-trained models: native RGB environment versus masked environment.}",
            r"\centering",
            dataframe_to_tabularx(rgb_df, font_size=r"\footnotesize", column_sep=4),
            r"\end{table}",
            r"\subsection{Grayscale Models}",
            r"\begin{table}[H]",
            r"\caption{Grayscale-trained models: native grayscale environment versus masked environment.}",
            r"\centering",
            dataframe_to_tabularx(grayscale_df, font_size=r"\footnotesize", column_sep=4),
            r"\end{table}",
        ]
    )


def build_masked_transfer_section(comparison_df: pd.DataFrame) -> str:
    display_df = comparison_df[
        [
            "agent",
            "environment",
            "optimized_model",
            "masked_model",
            "mean_reward_optimized",
            "std_reward_optimized",
            "mean_reward_masked",
            "std_reward_masked",
            "mean_diff_optimized_minus_masked",
            "p_value_holm",
            "outcome",
        ]
    ].copy()
    display_df = display_df.rename(
        columns={
            "agent": "Agent",
            "environment": "Environment",
            "optimized_model": "Optimized model",
            "masked_model": "Masked model",
            "mean_reward_optimized": "Mean optimized",
            "std_reward_optimized": "Std optimized",
            "mean_reward_masked": "Mean masked",
            "std_reward_masked": "Std masked",
            "mean_diff_optimized_minus_masked": "Diff opt-masked",
            "p_value_holm": "Holm p-value",
            "outcome": "Outcome",
        }
    )
    for col in [
        "Mean optimized",
        "Std optimized",
        "Mean masked",
        "Std masked",
        "Diff opt-masked",
        "Holm p-value",
    ]:
        display_df[col] = display_df[col].map(lambda x: f"{float(x):.3f}")

    rgb_df = display_df[display_df["Environment"] == "rgb"].reset_index(drop=True)
    grayscale_df = display_df[display_df["Environment"] == "grayscale"].reset_index(drop=True)

    return "\n".join(
        [
            r"\section{Masked Model vs Environment-Optimized Model}",
            r"This section tests how the masked-trained model performs in RGB and grayscale environments relative to the model optimized for that same evaluation environment.",
            r"\subsection{RGB Environment}",
            r"\begin{table}[H]",
            r"\caption{Masked-trained models versus RGB-optimized models, evaluated in the RGB environment.}",
            r"\centering",
            dataframe_to_tabularx(rgb_df, font_size=r"\footnotesize", column_sep=4),
            r"\end{table}",
            r"\subsection{Grayscale Environment}",
            r"\begin{table}[H]",
            r"\caption{Masked-trained models versus grayscale-optimized models, evaluated in the grayscale environment.}",
            r"\centering",
            dataframe_to_tabularx(grayscale_df, font_size=r"\footnotesize", column_sep=4),
            r"\end{table}",
        ]
    )


def build_explanation_section() -> str:
    return "\n".join(
        [
            r"\section{How to Read This Report}",
            r"This report keeps only the sections most relevant to transfer toward the masked setting and from the masked model toward other environments.",
            r"\begin{itemize}",
            r"\item \textbf{Masked Environment}: full ranking of all six best-trial models when evaluation is performed directly in the masked environment, including descriptive statistics, omnibus test, and pairwise outcomes.",
            r"\item \textbf{Own Environment vs Masked}: direct paired comparison for RGB-optimized and grayscale-optimized models between their native environment and the masked environment, so the table shows how much performance changes when the same model is moved to masked evaluation.",
            r"\item \textbf{Masked Model vs Environment-Optimized Model}: direct paired comparison in RGB and grayscale environments between the masked-trained model and the model optimized for that same environment, so the table shows whether the masked version transfers competitively outside its native setting.",
            r"\end{itemize}",
        ]
    )


def build_document(title: str, sections: list[str]) -> str:
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage[table]{{xcolor}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{microtype}}
\usepackage{{hyperref}}
\usepackage{{parskip}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\setcounter{{secnumdepth}}{{3}}
\begin{{document}}
\title{{{latex_escape(title)}}}
\date{{}}
\maketitle
Best-trial comparison rebuilt from \texttt{{models\_checkpoints}}, grouped by evaluation environment and compared on seed-level reward statistics.
{chr(10).join(sections)}
\end{{document}}
"""


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_df, seed_df = load_all_best_trial_eval(input_dir)
    summary_df = build_environment_summary(seed_df, episode_df)
    omnibus_df, pairwise_df = build_environment_tests(seed_df)
    own_vs_masked_df = build_own_vs_masked_summary(seed_df)
    masked_transfer_df = build_masked_transfer_summary(seed_df)

    episode_df.to_csv(output_dir / "episode_level_by_environment.csv", index=False)
    seed_df.to_csv(output_dir / "seed_level_by_environment.csv", index=False)
    summary_df.to_csv(output_dir / "environment_summary.csv", index=False)
    omnibus_df.to_csv(output_dir / "environment_omnibus.csv", index=False)
    pairwise_df.to_csv(output_dir / "environment_pairwise.csv", index=False)
    own_vs_masked_df.to_csv(output_dir / "own_vs_masked_comparison.csv", index=False)
    masked_transfer_df.to_csv(output_dir / "masked_vs_optimized_comparison.csv", index=False)

    sections = [
        build_explanation_section(),
        build_section(summary_df, omnibus_df, pairwise_df, "masked"),
        build_own_vs_masked_section(own_vs_masked_df),
        build_masked_transfer_section(masked_transfer_df),
    ]
    tex = build_document(args.title, sections)
    tex_path = output_dir / "best_trial_env_tables.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(tex_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
