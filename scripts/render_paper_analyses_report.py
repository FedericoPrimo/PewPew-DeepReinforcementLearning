from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats


LABEL_MAP = {
    "Best-RGB-optimized/ppo": "PPO-RGB",
    "Best-Greyscale-optimized/ppo": "PPO-GRY",
    "Best-Masked-optimized/ppo": "PPO-MSK",
    "Best-RGB-optimized/dqn": "DQN-RGB",
    "Best-Greyscale-optimized/dqn": "DQN-GRY",
    "Best-Masked-optimized/dqn": "DQN-MSK",
}

TRAIN_ENV_ORDER = ["rgb", "grayscale", "masked"]
TEST_ENV_ORDER_FULL = ["rgb", "grayscale", "masked"]
ENV_LABELS = {"rgb": "RGB", "grayscale": "Greyscale", "masked": "Masked"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a LaTeX/PDF report for the paper analyses bundle.")
    parser.add_argument("--input-dir", default="results/paper_analyses")
    parser.add_argument("--output-dir", default="results/paper_analyses/report")
    parser.add_argument("--title", default="Extended Analysis Report v3")
    parser.add_argument("--models-dir", default="models_checkpoints")
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


def dataframe_to_tabular(df: pd.DataFrame, font_size: str = r"\footnotesize", column_sep: int = 5) -> str:
    first_col_width = 0.24
    remaining_width = 0.72
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


def build_failure_table(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "failure" / "failure_summary.csv")
    table = df[
        [
            "display_label",
            "mean_reward",
            "std_reward",
            "failed_episode_pct",
            "low_reward_pct",
        ]
    ].copy()
    table = table.rename(
        columns={
            "display_label": "Agent",
            "mean_reward": "Mean reward",
            "std_reward": "Std reward",
            "failed_episode_pct": "Failed %",
            "low_reward_pct": "Low-reward %",
        }
    )
    for column in ["Mean reward", "Std reward", "Failed %", "Low-reward %"]:
        table[column] = table[column].map(lambda x: f"{float(x):.2f}")
    return table


def build_efficiency_table(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "efficiency" / "efficiency_summary.csv")
    table = df[
        [
            "display_label",
            "mean_reward",
            "training_time_minutes",
            "mean_steps_per_episode",
            "reward_per_minute",
        ]
    ].copy()
    table = table.rename(
        columns={
            "display_label": "Agent",
            "mean_reward": "Mean reward",
            "training_time_minutes": "Training time (min)",
            "mean_steps_per_episode": "Mean steps / episode",
            "reward_per_minute": "Reward / min",
        }
    )
    for column in ["Mean reward", "Training time (min)", "Mean steps / episode", "Reward / min"]:
        table[column] = table[column].map(lambda x: f"{float(x):.2f}")
    return table


def build_generalisation_table(input_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(input_dir / "generalisation" / "generalisation_longform.csv")
    table = df[
        [
            "algorithm",
            "train_env_label",
            "test_env_label",
            "mean_reward",
            "std_reward",
            "outcome",
        ]
    ].copy()
    table = table.rename(
        columns={
            "algorithm": "Algo",
            "train_env_label": "Train env",
            "test_env_label": "Test env",
            "mean_reward": "Mean reward",
            "std_reward": "Std reward",
            "outcome": "Outcome",
        }
    )
    for column in ["Mean reward", "Std reward"]:
        table[column] = table[column].map(lambda x: f"{float(x):.2f}")
    return table


def short_label_for(train_env: str, algorithm: str) -> str:
    if train_env == "rgb":
        return f"Best-RGB-optimized/{algorithm.lower()}"
    if train_env == "grayscale":
        return f"Best-Greyscale-optimized/{algorithm.lower()}"
    if train_env == "masked":
        return f"Best-Masked-optimized/{algorithm.lower()}"
    raise ValueError(f"Unsupported environment: {train_env}")


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def load_cross_environment_seed_data(models_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for json_path in sorted(models_dir.glob("Best-*-optimized/*/best_trial/*_model_eval_*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        checkpoint_parts = Path(data["checkpoint"].replace("\\", "/")).parts
        model_folder = next(part for part in checkpoint_parts if part.startswith("Best-"))
        short_label = f"{model_folder}/{str(data['agent']).lower()}"
        display_label = LABEL_MAP[short_label]
        for seed_block in data["per_seed"]:
            rewards = [float(value) for value in seed_block["episode_rewards"]]
            rows.append(
                {
                    "short_label": short_label,
                    "display_label": display_label,
                    "algorithm": display_label.split("-")[0],
                    "test_env": data["setting"],
                    "seed": int(seed_block["seed"]),
                    "mean_reward_seed": float(np.mean(rewards)),
                }
            )
    return pd.DataFrame(rows)


def build_generalisation_summary_full(models_dir: Path) -> pd.DataFrame:
    cross_seed_df = load_cross_environment_seed_data(models_dir)
    rows: list[dict] = []
    for algorithm in ["PPO", "DQN"]:
        for train_env in TRAIN_ENV_ORDER:
            candidate_short = short_label_for(train_env, algorithm)
            candidate_display = LABEL_MAP[candidate_short]
            for test_env in TEST_ENV_ORDER_FULL:
                benchmark_short = short_label_for(test_env, algorithm)
                benchmark_display = LABEL_MAP[benchmark_short]

                candidate = cross_seed_df[
                    (cross_seed_df["short_label"] == candidate_short) & (cross_seed_df["test_env"] == test_env)
                ][["seed", "mean_reward_seed"]].rename(columns={"mean_reward_seed": "candidate_reward"})
                benchmark = cross_seed_df[
                    (cross_seed_df["short_label"] == benchmark_short) & (cross_seed_df["test_env"] == test_env)
                ][["seed", "mean_reward_seed"]].rename(columns={"mean_reward_seed": "benchmark_reward"})
                paired = candidate.merge(benchmark, on="seed", how="inner").sort_values("seed")

                candidate_values = paired["candidate_reward"].to_numpy()
                benchmark_values = paired["benchmark_reward"].to_numpy()
                diffs = candidate_values - benchmark_values

                if candidate_short == benchmark_short:
                    p_value = 1.0
                elif len(diffs) == 0 or np.allclose(diffs, 0.0):
                    p_value = 1.0
                else:
                    _, p_value = stats.wilcoxon(
                        candidate_values,
                        benchmark_values,
                        zero_method="wilcox",
                        alternative="two-sided",
                    )

                rows.append(
                    {
                        "algorithm": algorithm,
                        "train_env": train_env,
                        "train_env_label": ENV_LABELS[train_env],
                        "test_env": test_env,
                        "test_env_label": ENV_LABELS[test_env],
                        "candidate_short_label": candidate_short,
                        "candidate_display_label": candidate_display,
                        "benchmark_short_label": benchmark_short,
                        "benchmark_display_label": benchmark_display,
                        "is_environment_matched": candidate_short == benchmark_short,
                        "mean_reward": float(np.mean(candidate_values)),
                        "std_reward": float(np.std(candidate_values, ddof=1)) if len(candidate_values) > 1 else 0.0,
                        "mean_diff_vs_benchmark": float(np.mean(diffs)) if len(diffs) else 0.0,
                        "p_value_raw": float(p_value),
                    }
                )

    summary = pd.DataFrame(rows)
    adjusted_values = pd.Series(1.0, index=summary.index, dtype=float)
    for algorithm in ["PPO", "DQN"]:
        mask = (summary["algorithm"] == algorithm) & (
            summary["candidate_short_label"] != summary["benchmark_short_label"]
        )
        adjusted_values.loc[mask] = holm_adjust(summary.loc[mask, "p_value_raw"].tolist())
    summary["p_value_holm"] = adjusted_values
    summary["outcome"] = "Draw"
    summary.loc[
        (summary["candidate_short_label"] != summary["benchmark_short_label"])
        & (summary["p_value_holm"] < 0.05)
        & (summary["mean_diff_vs_benchmark"] > 0.0),
        "outcome",
    ] = "Win"
    summary.loc[
        (summary["candidate_short_label"] != summary["benchmark_short_label"])
        & (summary["p_value_holm"] < 0.05)
        & (summary["mean_diff_vs_benchmark"] < 0.0),
        "outcome",
    ] = "Loss"
    summary["highlight_masked_win"] = (summary["train_env"] == "masked") & (summary["outcome"] == "Win")
    return summary


def mean_std_text(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.2f} +/- {std_value:.2f}"


def create_generalisation_matrix_figures_with_masked(models_dir: Path, input_dir: Path, suffix: str) -> dict[str, str]:
    summary = build_generalisation_summary_full(models_dir)
    output_dir = input_dir / "generalisation"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, str] = {}

    for algorithm in ["PPO", "DQN"]:
        subset = summary[summary["algorithm"] == algorithm].copy()
        heat_values = (
            subset.pivot(index="train_env_label", columns="test_env_label", values="mean_reward")
            .reindex(index=[ENV_LABELS[env] for env in TRAIN_ENV_ORDER], columns=[ENV_LABELS[env] for env in TEST_ENV_ORDER_FULL])
        )
        annotation_values = (
            subset.assign(
                annotation=subset.apply(
                    lambda row: (
                        f"{mean_std_text(row['mean_reward'], row['std_reward'])}\n"
                        f"{'†' if row['is_environment_matched'] else row['outcome'] + (' *' if row['highlight_masked_win'] else '')}"
                    ),
                    axis=1,
                )
            )
            .pivot(index="train_env_label", columns="test_env_label", values="annotation")
            .reindex(index=[ENV_LABELS[env] for env in TRAIN_ENV_ORDER], columns=[ENV_LABELS[env] for env in TEST_ENV_ORDER_FULL])
        )
        highlight_matrix = (
            subset.pivot(index="train_env_label", columns="test_env_label", values="highlight_masked_win")
            .reindex(index=[ENV_LABELS[env] for env in TRAIN_ENV_ORDER], columns=[ENV_LABELS[env] for env in TEST_ENV_ORDER_FULL])
            .fillna(False)
        )

        fig, ax = plt.subplots(figsize=(10.0, 5.6))
        image = ax.imshow(heat_values.to_numpy(dtype=float), cmap="Blues", aspect="auto")
        cbar = fig.colorbar(image, ax=ax, shrink=0.92)
        cbar.set_label("Mean reward")

        ax.set_xticks(np.arange(len(heat_values.columns)))
        ax.set_xticklabels(heat_values.columns)
        ax.set_yticks(np.arange(len(heat_values.index)))
        ax.set_yticklabels(heat_values.index)
        ax.set_xlabel("Test environment")
        ax.set_ylabel("Training environment")
        ax.set_title(f"{algorithm} cross-environment evaluation matrix")

        for row_idx in range(len(heat_values.index)):
            for col_idx in range(len(heat_values.columns)):
                ax.text(
                    col_idx,
                    row_idx,
                    annotation_values.iloc[row_idx, col_idx],
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    color="#111111",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "alpha": 0.82,
                        "edgecolor": "#2f2f2f",
                        "linewidth": 0.4,
                    },
                )
                if bool(highlight_matrix.iloc[row_idx, col_idx]):
                    rect = mpatches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#2b8a3e",
                        linewidth=3.0,
                    )
                    ax.add_patch(rect)

        ax.set_xticks(np.arange(-0.5, len(heat_values.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(heat_values.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.text(
            0.0,
            -0.14,
            "* masked-trained model outperforms the environment-matched benchmark",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.3,
            color="#2b8a3e",
        )
        ax.text(
            0.0,
            -0.22,
            "† environment-matched baseline cell",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.3,
            color="#444444",
        )
        fig.tight_layout()

        stem = output_dir / f"generalisation_matrix_{algorithm.lower()}_with_masked{suffix}"
        for extension in ("png", "pdf"):
            fig.savefig(stem.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_paths[algorithm] = str(stem.with_suffix(".png"))

    return figure_paths


def figure_block(image_path: str, caption: str, width: str = r"0.92\linewidth") -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{image_path}}}",
            rf"\caption{{{caption}}}",
            r"\end{figure}",
        ]
    )


def build_document(
    title: str,
    failure_table: pd.DataFrame,
    efficiency_table: pd.DataFrame,
    generalisation_table: pd.DataFrame,
    ppo_generalisation_figure: str,
    dqn_generalisation_figure: str,
) -> str:
    generalisation_text = (
        "The following section reframes the transfer results as a training-environment by test-environment matrix, "
        "using the environment-matched optimized model as the paired benchmark. Cells marked with an asterisk "
        "indicate masked-trained models that outperform the environment-matched baseline."
    )
    generalisation_text += " The PPO and DQN matrices include Masked as a test environment, and the dagger symbol marks the environment-matched baseline cell."

    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage[table]{{xcolor}}
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

This report collects the additional analyses produced for the Atari Space Invaders study: failure analysis, efficiency analysis, and cross-environment generalisation.

\section{{Failure Analysis}}
Failure is defined using the 10th percentile of PPO-RGB episode rewards in its native evaluation environment. The accompanying plot also highlights low-reward episodes with reward strictly below 3.

\begin{{table}}[H]
\centering
\caption{{Failure summary across the six best-trial agents.}}
{dataframe_to_tabular(failure_table)}
\end{{table}}

{figure_block("results/paper_analyses/failure/episode_reward_distribution.png", "Distribution of episode rewards per agent. Red points mark low-reward episodes (reward $< 3$).")}

{figure_block("results/paper_analyses/failure/reward_over_episode_index.png", "Reward over evaluation episode index, split by algorithm, with thin seed-level traces and bold seed-average traces.")}

\section{{Efficiency Analysis}}
Training efficiency combines evaluation performance with training time and mean episode length reconstructed from the best-trial evaluations.

\begin{{table}}[H]
\centering
\caption{{Efficiency summary including reward, training time, average episode steps, and reward per minute.}}
{dataframe_to_tabular(efficiency_table)}
\end{{table}}

{figure_block("results/paper_analyses/efficiency/training_time_vs_reward.png", "Training time versus mean reward. Colors identify the algorithm and line styles identify the representation.")}

\section{{Cross-Environment Generalisation}}
{generalisation_text}

\begin{{table}}[H]
\centering
\caption{{Long-form summary of the cross-environment evaluation results.}}
{dataframe_to_tabular(generalisation_table, column_sep=4)}
\end{{table}}

{figure_block(ppo_generalisation_figure, "Cross-environment evaluation matrix for PPO.")}

{figure_block(dqn_generalisation_figure, "Cross-environment evaluation matrix for DQN.")}

\end{{document}}
"""


def compile_pdf(tex_path: Path, output_dir: Path) -> None:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={output_dir}",
        str(tex_path),
    ]
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_table = build_failure_table(input_dir)
    efficiency_table = build_efficiency_table(input_dir)
    generalisation_table = build_generalisation_table(input_dir)
    suffix = "_v3"
    figure_paths = create_generalisation_matrix_figures_with_masked(models_dir, input_dir, suffix)
    ppo_generalisation_figure = figure_paths["PPO"]
    dqn_generalisation_figure = figure_paths["DQN"]

    tex_content = build_document(
        args.title,
        failure_table,
        efficiency_table,
        generalisation_table,
        ppo_generalisation_figure,
        dqn_generalisation_figure,
    )
    base_name = f"paper_analyses_report{suffix}"
    tex_path = output_dir / f"{base_name}.tex"
    tex_path.write_text(tex_content, encoding="utf-8")

    compile_pdf(tex_path, output_dir)
    print(tex_path)
    print(output_dir / f"{base_name}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
