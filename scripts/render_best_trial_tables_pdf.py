from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un report LaTeX/PDF sintetico per i best trial.")
    parser.add_argument("--input-dir", default="results/best_trial_analysis")
    parser.add_argument("--output-dir", default="results/best_trial_analysis/report")
    parser.add_argument("--title", default="Best Trial Statistical Summary")
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


def build_summary_table(ranking_df: pd.DataFrame) -> pd.DataFrame:
    summary = ranking_df[["rank", "short_label", "mean_seed_reward", "std_seed_reward"]].copy()
    summary["short_label"] = summary["short_label"].map(lambda x: LABEL_MAP.get(x, x))
    summary = summary.rename(
        columns={
            "rank": "Rank",
            "short_label": "Model",
            "mean_seed_reward": "Mean reward",
            "std_seed_reward": "Std dev",
        }
    )
    summary["Mean reward"] = summary["Mean reward"].map(lambda x: f"{x:.3f}")
    summary["Std dev"] = summary["Std dev"].map(lambda x: f"{x:.3f}")
    return summary


def build_step_table(ranking_df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    episode_lengths = load_episode_length_samples(ranking_df, input_dir)
    step_summary = (
        episode_lengths.groupby(["run_id", "short_label"], as_index=False)
        .agg(
            mean_episode_steps=("episode_steps", "mean"),
            std_episode_steps=("episode_steps", "std"),
            n_selected_episodes=("episode_steps", "count"),
        )
    )

    rows = []
    for row in ranking_df.itertuples(index=False):
        summary_row = step_summary[step_summary["short_label"] == row.short_label].iloc[0]
        rows.append(
            {
                "Rank": row.rank,
                "Model": LABEL_MAP.get(row.short_label, row.short_label),
                "Mean steps": f"{float(summary_row['mean_episode_steps']):.2f}",
                "Std dev": f"{float(summary_row['std_episode_steps']):.2f}",
                "N": int(summary_row["n_selected_episodes"]),
            }
        )
    return pd.DataFrame(rows)


def ensure_all_episode_lengths(input_dir: Path) -> pd.DataFrame:
    cache_path = input_dir / "episode_length_all.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    analyze_path = Path(__file__).resolve().parent / "analyze_best_trials.py"
    spec = importlib.util.spec_from_file_location("analyze_best_trials", analyze_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile caricare {analyze_path}")
    aba = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = aba
    spec.loader.exec_module(aba)
    from src.envs.atari_wrappers import make_atari_env
    from src.envs.vec_env import make_eval_vec_env

    runs = aba.discover_best_runs(Path("models_checkpoints"))
    common_cfg = aba.load_config(aba.PROJECT_ROOT / "configs/config_common.yaml")
    rows: list[dict] = []

    for run in runs:
        with run.eval_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        per_seed = data.get("per_seed", [])
        preprocessing_kwargs, masking_kwargs = aba.build_setting_kwargs(run.setting)
        agent = aba.load_agent(run)

        if run.agent == "ppo":
            for seed_block in per_seed:
                seed = int(seed_block["seed"])
                episodes = len(seed_block.get("episode_rewards", []))
                aba.set_global_seed(seed)
                env = make_eval_vec_env(
                    common_cfg.env.id,
                    seed=seed,
                    preprocessing_kwargs=preprocessing_kwargs,
                    masking_kwargs=masking_kwargs,
                )
                try:
                    obs = env.reset()
                    for episode_index in range(1, episodes + 1):
                        done = False
                        step_idx = 0
                        while not done:
                            action = agent.act(np.asarray(obs)[0])
                            obs, _, done_arr, _ = env.step(np.array([action]))
                            step_idx += 1
                            done = bool(done_arr[0])
                        rows.append(
                            {
                                "run_id": run.run_id,
                                "short_label": run.short_label,
                                "seed": seed,
                                "episode_index": episode_index,
                                "episode_steps": step_idx,
                            }
                        )
                finally:
                    env.close()
        else:
            for seed_block in per_seed:
                seed = int(seed_block["seed"])
                episodes = len(seed_block.get("episode_rewards", []))
                aba.set_global_seed(seed)
                env = make_atari_env(
                    common_cfg.env.id,
                    seed=seed,
                    preprocessing_kwargs=preprocessing_kwargs,
                    masking_kwargs=masking_kwargs,
                )
                try:
                    for episode_index in range(1, episodes + 1):
                        obs, _ = env.reset()
                        done = False
                        step_idx = 0
                        while not done:
                            action = agent.act(np.asarray(obs))
                            obs, _, terminated, truncated, _ = env.step(action)
                            step_idx += 1
                            done = bool(terminated or truncated)
                        rows.append(
                            {
                                "run_id": run.run_id,
                                "short_label": run.short_label,
                                "seed": seed,
                                "episode_index": episode_index,
                                "episode_steps": step_idx,
                            }
                        )
                finally:
                    env.close()

    lengths = pd.DataFrame(rows)
    lengths.to_csv(cache_path, index=False)
    return lengths


def load_episode_length_samples(ranking_df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    lengths = ensure_all_episode_lengths(input_dir).copy()
    lengths["display_label"] = lengths["short_label"].map(lambda x: LABEL_MAP.get(x, x))
    order_map = {row.run_id: row.rank for row in ranking_df.itertuples(index=False)}
    lengths["rank"] = lengths["run_id"].map(order_map)
    return lengths.sort_values(["rank", "seed", "episode_index"]).reset_index(drop=True)


def build_outcome_matrix(ranking_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    models = [LABEL_MAP.get(label, label) for label in ranking_df["short_label"].tolist()]
    models = sorted(models, key=lambda label: (agent_group(label), models.index(label)))
    run_id_to_label = {
        run_id: LABEL_MAP.get(label, label)
        for run_id, label in zip(ranking_df["run_id"], ranking_df["short_label"])
    }
    matrix = pd.DataFrame("Pareggio", index=models, columns=models)

    for model in models:
        matrix.loc[model, model] = "-"

    for row in pairwise_df.itertuples(index=False):
        label_a = run_id_to_label.get(row.run_a)
        label_b = run_id_to_label.get(row.run_b)
        if label_a is None or label_b is None:
            continue

        if bool(row.significant_0_05):
            if row.mean_diff_a_minus_b > 0:
                matrix.loc[label_a, label_b] = "Vittoria"
                matrix.loc[label_b, label_a] = "Sconfitta"
            elif row.mean_diff_a_minus_b < 0:
                matrix.loc[label_a, label_b] = "Sconfitta"
                matrix.loc[label_b, label_a] = "Vittoria"
            else:
                matrix.loc[label_a, label_b] = "Pareggio"
                matrix.loc[label_b, label_a] = "Pareggio"
        else:
            matrix.loc[label_a, label_b] = "Pareggio"
            matrix.loc[label_b, label_a] = "Pareggio"

    matrix.insert(0, "Model", matrix.index)
    matrix = matrix.reset_index(drop=True)
    return matrix


def build_outcome_summary(matrix_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in matrix_df.itertuples(index=False):
        values = [value for value in row[1:] if value != "-"]
        rows.append(
            {
                "Model": row[0],
                "Vittorie": sum(value == "Vittoria" for value in values),
                "Pareggi": sum(value == "Pareggio" for value in values),
                "Sconfitte": sum(value == "Sconfitta" for value in values),
            }
        )
    return pd.DataFrame(rows)


def build_step_outcome_matrix(ranking_df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    lengths = load_episode_length_samples(ranking_df, input_dir)
    models = [LABEL_MAP.get(label, label) for label in ranking_df["short_label"].tolist()]
    models = sorted(models, key=lambda label: (agent_group(label), models.index(label)))
    matrix = pd.DataFrame("Pareggio", index=models, columns=models)

    for model in models:
        matrix.loc[model, model] = "-"

    for left_index, left_model in enumerate(models):
        left_rows = lengths[lengths["display_label"] == left_model].sort_values(["seed", "episode_index"])
        left_values = left_rows["episode_steps"].to_numpy()

        for right_model in models[left_index + 1:]:
            right_rows = lengths[lengths["display_label"] == right_model].sort_values(["seed", "episode_index"])
            right_values = right_rows["episode_steps"].to_numpy()
            diffs = left_values - right_values

            if len(left_values) != len(right_values) or len(left_values) == 0:
                outcome_left = "Pareggio"
                outcome_right = "Pareggio"
            elif bool((diffs == 0).all()):
                outcome_left = "Pareggio"
                outcome_right = "Pareggio"
            else:
                statistic, p_value = stats.wilcoxon(left_values, right_values, zero_method="wilcox", alternative="two-sided")
                if p_value < 0.05:
                    if diffs.mean() > 0:
                        outcome_left = "Vittoria"
                        outcome_right = "Sconfitta"
                    elif diffs.mean() < 0:
                        outcome_left = "Sconfitta"
                        outcome_right = "Vittoria"
                    else:
                        outcome_left = "Pareggio"
                        outcome_right = "Pareggio"
                else:
                    outcome_left = "Pareggio"
                    outcome_right = "Pareggio"

            matrix.loc[left_model, right_model] = outcome_left
            matrix.loc[right_model, left_model] = outcome_right

    matrix.insert(0, "Model", matrix.index)
    return matrix.reset_index(drop=True)


def dataframe_to_tabularx(df: pd.DataFrame, font_size: str = r"\small", column_sep: int = 6) -> str:
    first_col_width = 0.20 if len(df.columns) <= 5 else 0.16
    remaining_width = max(0.78, 0.98 - first_col_width)
    col_width = remaining_width / max(1, len(df.columns) - 1)
    spec = (
        f">{{\\raggedright\\arraybackslash}}p{{{first_col_width:.2f}\\linewidth}}"
        + f">{{\\centering\\arraybackslash}}p{{{col_width:.2f}\\linewidth}}" * (len(df.columns) - 1)
    )
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

    first_col_width = 0.16
    remaining_width = 0.82
    col_width = remaining_width / max(1, len(columns) - 1)
    spec_parts = [f">{{\\raggedright\\arraybackslash}}p{{{first_col_width:.2f}\\linewidth}}"]
    for idx in range(1, len(columns)):
        prefix = "|" if idx == ppo_count + 1 else ""
        spec_parts.append(
            f"{prefix}>{{\\centering\\arraybackslash}}p{{{col_width:.2f}\\linewidth}}"
        )
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


def outcome_summary_to_list(df: pd.DataFrame, title: str) -> str:
    lines = [
        rf"\noindent\textbf{{{latex_escape(title)}}}",
        r"\begin{itemize}",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            rf"\item {latex_escape(row.Model)}: "
            rf"{row.Vittorie} vittorie, {row.Pareggi} pareggi, {row.Sconfitte} sconfitte."
        )
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def build_episode_figure_block(input_dir: Path) -> str:
    selected_df = pd.read_csv(input_dir / "selected_episode_regimes.csv")
    figure_lines = [
        r"\begin{figure}[H]",
        r"\centering",
    ]
    regime_titles = {
        "optimistic": "Optimistic cases",
        "medium": "Average cases",
        "pessimistic": "Pessimistic cases",
    }
    for regime in ["optimistic", "medium", "pessimistic"]:
        rows = list(selected_df[selected_df["regime"] == regime].itertuples(index=False))
        if not rows:
            continue
        figure_lines.append(rf"\noindent\textbf{{{regime_titles[regime]}}}\\[2pt]")
        for index, row in enumerate(rows):
            path = f"results/best_trial_analysis/plots/temporal_{row.regime}_seed_{row.seed}_episode_{row.episode_index}.png"
            caption = f"seed={row.seed}, episode={row.episode_index}"
            figure_lines.extend(
                [
                    r"\begin{minipage}[t]{0.32\textwidth}",
                    r"\centering",
                    rf"\includegraphics[width=\linewidth]{{{path}}}",
                    rf"\caption*{{{latex_escape(caption)}}}",
                    r"\end{minipage}",
                ]
            )
            if index < len(rows) - 1:
                figure_lines.append(r"\hfill")
        figure_lines.append(r"\vspace{0.4cm}")
    figure_lines.extend(
        [
            r"\caption{Temporal reward trajectories grouped into optimistic, average, and pessimistic cases.}",
            r"\end{figure}",
        ]
    )
    return "\n".join(figure_lines)


def build_document(
    title: str,
    summary_df: pd.DataFrame,
    step_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    outcome_summary_df: pd.DataFrame,
    step_matrix_df: pd.DataFrame,
    step_outcome_summary_df: pd.DataFrame,
    episode_figures_tex: str,
) -> str:
    summary_table = dataframe_to_tabularx(summary_df, font_size=r"\footnotesize", column_sep=6)
    steps_table = dataframe_to_tabularx(step_df, font_size=r"\footnotesize", column_sep=6)
    matrix_table = pairwise_matrix_to_tabular(matrix_df, font_size=r"\scriptsize", column_sep=3)
    step_matrix_table = pairwise_matrix_to_tabular(step_matrix_df, font_size=r"\scriptsize", column_sep=3)
    reward_outcome_list = outcome_summary_to_list(outcome_summary_df, "Reward outcome counts")
    step_outcome_list = outcome_summary_to_list(step_outcome_summary_df, "Episode-step outcome counts")

    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=0.7in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage[table]{{xcolor}}
\usepackage{{caption}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{microtype}}
\usepackage{{hyperref}}
\usepackage{{parskip}}
\hypersetup{{colorlinks=true,linkcolor=black,urlcolor=blue}}
\begin{{document}}
\title{{{latex_escape(title)}}}
\date{{}}
\maketitle
Best-trial summary rebuilt from \texttt{{results/best\_trial\_analysis}}, with reward ranking, episode-length statistics, pairwise outcomes, and representative temporal trajectories.

\begin{{table}}[H]
\caption{{Summary statistics of the best trials.}}
\centering
\begin{{minipage}}[t]{{0.64\linewidth}}
\captionsetup{{type=table}}
\caption*{{\textbf{{(a)}} Mean reward and standard deviation}}
{summary_table}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.34\linewidth}}
\captionsetup{{type=table}}
\caption*{{\textbf{{(b)}} Episode length over all evaluation episodes}}
{steps_table}
\end{{minipage}}
\end{{table}}

\begin{{table}}[H]
\caption{{Pairwise outcomes for reward and episode length.}}
\centering
\begin{{minipage}}[t]{{0.49\linewidth}}
\captionsetup{{type=table}}
\caption*{{\textbf{{(a)}} Reward outcomes}}
{matrix_table}
{reward_outcome_list}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\linewidth}}
\captionsetup{{type=table}}
\caption*{{\textbf{{(b)}} Episode-length outcomes}}
{step_matrix_table}
{step_outcome_list}
\end{{minipage}}
\end{{table}}

{episode_figures_tex}

\end{{document}}
"""


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_df = pd.read_csv(input_dir / "ranking.csv")
    pairwise_df = pd.read_csv(input_dir / "pairwise_wilcoxon.csv")

    summary_df = build_summary_table(ranking_df)
    step_df = build_step_table(ranking_df, input_dir)
    matrix_df = build_outcome_matrix(ranking_df, pairwise_df)
    outcome_summary_df = build_outcome_summary(matrix_df)
    step_matrix_df = build_step_outcome_matrix(ranking_df, input_dir)
    step_outcome_summary_df = build_outcome_summary(step_matrix_df)
    episode_figures_tex = build_episode_figure_block(input_dir)

    tex = build_document(
        args.title,
        summary_df,
        step_df,
        matrix_df,
        outcome_summary_df,
        step_matrix_df,
        step_outcome_summary_df,
        episode_figures_tex,
    )
    tex_path = output_dir / "best_trial_tables.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(tex_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
