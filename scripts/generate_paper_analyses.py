from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

OWN_SETTING_BY_MODEL = {
    "Best-RGB-optimized": "rgb",
    "Best-Greyscale-optimized": "grayscale",
    "Best-Grayscale-optimized": "grayscale",
    "Best-Masked-optimized": "masked",
}

DISPLAY_ORDER = ["PPO-RGB", "PPO-GRY", "PPO-MSK", "DQN-RGB", "DQN-GRY", "DQN-MSK"]
TRAIN_ENV_ORDER = ["rgb", "grayscale", "masked"]
TEST_ENV_ORDER = ["rgb", "grayscale"]
ENV_LABELS = {"rgb": "RGB", "grayscale": "Greyscale", "masked": "Masked"}
ALGO_COLORS = {"PPO": "#1f77b4", "DQN": "#ff7f0e"}
REP_LINESTYLES = {"RGB": "-", "GRY": "--", "MSK": ":"}
REP_MARKERS = {"RGB": "o", "GRY": "s", "MSK": "^"}
TRAINING_TIME_MINUTES = {
    "DQN-GRY": 38.97,
    "DQN-MSK": 24.46,
    "DQN-RGB": 32.62,
    "PPO-GRY": 32.51,
    "PPO-MSK": 30.72,
    "PPO-RGB": 38.42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready failure, efficiency, and generalisation analyses.")
    parser.add_argument("--input-dir", default="models_checkpoints", help="Base directory containing best-trial checkpoints.")
    parser.add_argument("--output-dir", default="results/paper_analyses", help="Directory where tables and figures will be saved.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
        }
    )


def display_to_short_label(display_label: str) -> str:
    inverse = {value: key for key, value in LABEL_MAP.items()}
    return inverse[display_label]


def short_label_for(train_env: str, algorithm: str) -> str:
    if train_env == "rgb":
        return f"Best-RGB-optimized/{algorithm.lower()}"
    if train_env == "grayscale":
        return f"Best-Greyscale-optimized/{algorithm.lower()}"
    if train_env == "masked":
        return f"Best-Masked-optimized/{algorithm.lower()}"
    raise ValueError(f"Unsupported environment: {train_env}")


def algorithm_of(display_label: str) -> str:
    return display_label.split("-")[0]


def representation_of(display_label: str) -> str:
    return display_label.split("-")[1]


def mean_std_text(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.2f} +/- {std_value:.2f}"


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


def load_own_environment_episode_data(input_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for model_label, own_setting in OWN_SETTING_BY_MODEL.items():
        model_dir = input_dir / model_label
        if not model_dir.exists():
            continue
        for agent_dir in sorted(model_dir.iterdir()):
            best_trial_dir = agent_dir / "best_trial"
            agent = agent_dir.name.lower()
            eval_path = best_trial_dir / f"{agent}_model_eval_{own_setting}.json"
            if not eval_path.exists():
                continue
            with eval_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            short_label = f"{model_label}/{agent}"
            display_label = LABEL_MAP[short_label]
            for seed_block in data["per_seed"]:
                seed = int(seed_block["seed"])
                rewards = [float(value) for value in seed_block["episode_rewards"]]
                for episode_index, reward in enumerate(rewards, start=1):
                    rows.append(
                        {
                            "short_label": short_label,
                            "display_label": display_label,
                            "algorithm": algorithm_of(display_label),
                            "representation": representation_of(display_label),
                            "setting": own_setting,
                            "seed": seed,
                            "episode_index": episode_index,
                            "reward": reward,
                        }
                    )
    df = pd.DataFrame(rows)
    display_order_map = {label: idx for idx, label in enumerate(DISPLAY_ORDER)}
    return df.sort_values(["display_label", "seed", "episode_index"], key=lambda col: col.map(display_order_map) if col.name == "display_label" else col).reset_index(drop=True)


def load_cross_environment_seed_data(input_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for json_path in sorted(input_dir.glob("Best-*-optimized/*/best_trial/*_model_eval_*.json")):
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
                    "algorithm": algorithm_of(display_label),
                    "representation": representation_of(display_label),
                    "test_env": data["setting"],
                    "seed": int(seed_block["seed"]),
                    "mean_reward_seed": float(np.mean(rewards)),
                    "std_reward_seed": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
                    "episodes_in_seed": len(rewards),
                }
            )
    return pd.DataFrame(rows)


def load_episode_lengths() -> pd.DataFrame:
    cache_path = Path("results/best_trial_analysis/episode_length_all.csv")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Episode length cache not found at {cache_path}. Run scripts/render_best_trial_tables_pdf.py first."
        )
    return pd.read_csv(cache_path)


def export_dataframe_table_figure(
    df: pd.DataFrame,
    title: str,
    output_stem: Path,
    row_labels: list[str] | None = None,
) -> None:
    n_rows = len(df)
    fig_height = max(2.2, 0.52 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.3 * len(df.columns)), fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for col_idx in range(len(df.columns)):
        header_cell = table[0, col_idx]
        header_cell.set_facecolor("#e9eef5")
        header_cell.set_text_props(weight="bold")

    if row_labels is not None:
        row_index_map = {label: idx for idx, label in enumerate(row_labels, start=1)}
        for label, row_idx in row_index_map.items():
            algo = algorithm_of(label)
            rep = representation_of(label)
            for col_idx in range(len(df.columns)):
                cell = table[row_idx, col_idx]
                cell.set_edgecolor(ALGO_COLORS[algo])
                cell.set_linewidth(1.3)
                cell.set_linestyle(REP_LINESTYLES[rep])

    ax.set_title(title, pad=16)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def failure_analysis(episode_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, float]:
    failure_output_dir = output_dir / "failure"
    failure_output_dir.mkdir(parents=True, exist_ok=True)

    ppo_rgb_rewards = episode_df.loc[episode_df["display_label"] == "PPO-RGB", "reward"].to_numpy()
    failure_threshold = float(np.quantile(ppo_rgb_rewards, 0.10))

    summary = (
        episode_df.groupby(["display_label", "algorithm", "representation"], as_index=False)
        .agg(
            mean_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            median_reward=("reward", "median"),
            total_episodes=("reward", "size"),
        )
    )
    failure_counts = episode_df.assign(is_failed=episode_df["reward"] < failure_threshold).groupby("display_label")["is_failed"].sum()
    summary["failed_episodes"] = summary["display_label"].map(failure_counts).astype(int)
    summary["failed_episode_pct"] = 100.0 * summary["failed_episodes"] / summary["total_episodes"]
    low_reward_counts = episode_df.assign(is_low_reward=episode_df["reward"] < 3.0).groupby("display_label")["is_low_reward"].sum()
    summary["low_reward_episodes"] = summary["display_label"].map(low_reward_counts).astype(int)
    summary["low_reward_pct"] = 100.0 * summary["low_reward_episodes"] / summary["total_episodes"]
    order_map = {label: idx for idx, label in enumerate(DISPLAY_ORDER)}
    summary = summary.sort_values("display_label", key=lambda col: col.map(order_map)).reset_index(drop=True)
    summary.to_csv(failure_output_dir / "failure_summary.csv", index=False)

    table_df = summary[
        [
            "display_label",
            "mean_reward",
            "std_reward",
            "failed_episodes",
            "failed_episode_pct",
            "low_reward_episodes",
            "low_reward_pct",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "display_label": "Agent",
            "mean_reward": "Mean reward",
            "std_reward": "Std reward",
            "failed_episodes": f"Failed eps (< {failure_threshold:.1f})",
            "failed_episode_pct": "Failed %",
            "low_reward_episodes": "Low-reward eps (< 3)",
            "low_reward_pct": "Low-reward %",
        }
    )
    for column in ["Mean reward", "Std reward", "Failed %", "Low-reward %"]:
        table_df[column] = table_df[column].map(lambda value: f"{float(value):.2f}")
    export_dataframe_table_figure(
        table_df,
        title=f"Failure summary (failure threshold = PPO-RGB 10th percentile = {failure_threshold:.2f})",
        output_stem=failure_output_dir / "failure_summary_table",
        row_labels=summary["display_label"].tolist(),
    )

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    positions = np.arange(1, len(DISPLAY_ORDER) + 1)
    reward_groups = [
        episode_df.loc[episode_df["display_label"] == display_label, "reward"].to_numpy()
        for display_label in DISPLAY_ORDER
    ]

    box = ax.boxplot(
        reward_groups,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1c1c1c", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    for patch, display_label in zip(box["boxes"], DISPLAY_ORDER):
        algo = algorithm_of(display_label)
        rep = representation_of(display_label)
        patch.set_facecolor(ALGO_COLORS[algo])
        patch.set_alpha(0.22)
        patch.set_edgecolor(ALGO_COLORS[algo])
        patch.set_linewidth(2.0)
        patch.set_linestyle(REP_LINESTYLES[rep])

    rng = np.random.default_rng(7)
    low_reward_label_used = False
    for position, display_label in zip(positions, DISPLAY_ORDER):
        subset = episode_df[episode_df["display_label"] == display_label].copy()
        x = position + rng.uniform(-0.16, 0.16, size=len(subset))
        normal_mask = subset["reward"] >= 3.0
        ax.scatter(
            x[normal_mask.to_numpy()],
            subset.loc[normal_mask, "reward"],
            s=16,
            color=ALGO_COLORS[algorithm_of(display_label)],
            alpha=0.18,
            linewidths=0.0,
        )
        label = "Low-reward episodes (< 3)" if not low_reward_label_used else None
        ax.scatter(
            x[(~normal_mask).to_numpy()],
            subset.loc[~normal_mask, "reward"],
            s=28,
            color="#c92a2a",
            alpha=0.75,
            linewidths=0.4,
            edgecolors="white",
            label=label,
            zorder=5,
        )
        low_reward_label_used = True

    ax.axhline(3.0, color="#c92a2a", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(failure_threshold, color="#2b2b2b", linestyle=":", linewidth=1.4, alpha=0.95)
    ax.text(
        0.55,
        failure_threshold + 0.25,
        f"Failure threshold (PPO-RGB p10) = {failure_threshold:.2f}",
        fontsize=9,
        color="#2b2b2b",
    )
    ax.set_xlim(0.35, len(DISPLAY_ORDER) + 0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels(DISPLAY_ORDER)
    ax.set_ylabel("Episode reward")
    ax.set_xlabel("Agent")
    ax.set_title("Episode reward distributions with low-reward episodes highlighted")

    algo_handles = [
        mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.22, label=f"{algo} color")
        for algo, color in ALGO_COLORS.items()
    ]
    rep_handles = [
        mlines.Line2D([0], [0], color="#555555", linestyle=linestyle, linewidth=2.0, label=f"{rep} style")
        for rep, linestyle in REP_LINESTYLES.items()
    ]
    extra_handles = [
        mlines.Line2D([0], [0], color="#c92a2a", marker="o", linestyle="None", markersize=7, label="Low-reward episodes (< 3)"),
        mlines.Line2D([0], [0], color="#2b2b2b", linestyle=":", linewidth=1.4, label="Failure threshold"),
    ]
    ax.legend(handles=algo_handles + rep_handles + extra_handles, ncol=2, frameon=True, loc="upper right")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(failure_output_dir / f"episode_reward_distribution.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, algorithm in zip(axes, ["PPO", "DQN"]):
        subset = episode_df[episode_df["algorithm"] == algorithm].copy()
        for display_label in [label for label in DISPLAY_ORDER if label.startswith(algorithm)]:
            label_subset = subset[subset["display_label"] == display_label].copy()
            for _, seed_frame in label_subset.groupby("seed"):
                ax.plot(
                    seed_frame["episode_index"],
                    seed_frame["reward"],
                    color=ALGO_COLORS[algorithm],
                    linestyle=REP_LINESTYLES[representation_of(display_label)],
                    linewidth=0.8,
                    alpha=0.12,
                )
            mean_trace = (
                label_subset.groupby("episode_index", as_index=False)
                .agg(mean_reward=("reward", "mean"), std_reward=("reward", "std"), n=("reward", "size"))
            )
            sem = mean_trace["std_reward"].fillna(0.0) / np.sqrt(mean_trace["n"].clip(lower=1))
            ax.plot(
                mean_trace["episode_index"],
                mean_trace["mean_reward"],
                color=ALGO_COLORS[algorithm],
                linestyle=REP_LINESTYLES[representation_of(display_label)],
                linewidth=2.4,
                label=display_label,
            )
            ax.fill_between(
                mean_trace["episode_index"],
                mean_trace["mean_reward"] - sem,
                mean_trace["mean_reward"] + sem,
                color=ALGO_COLORS[algorithm],
                alpha=0.08,
            )
        ax.set_title(f"{algorithm}: reward over evaluation episode index")
        ax.set_xlabel("Episode index within seed")
        ax.set_xlim(1, 30)
        ax.legend(frameon=True, loc="upper right")
    axes[0].set_ylabel("Episode reward")
    fig.suptitle("Evaluation stability across seeds (thin lines = individual seeds, bold lines = seed mean)", y=1.02)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(failure_output_dir / f"reward_over_episode_index.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary, failure_threshold


def efficiency_analysis(episode_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    efficiency_output_dir = output_dir / "efficiency"
    efficiency_output_dir.mkdir(parents=True, exist_ok=True)

    episode_lengths = load_episode_lengths()
    mean_reward_by_agent = episode_df.groupby("display_label", as_index=False)["reward"].mean().rename(columns={"reward": "mean_reward"})
    mean_steps_by_agent = (
        episode_lengths.assign(display_label=episode_lengths["short_label"].map(LABEL_MAP))
        .groupby("display_label", as_index=False)["episode_steps"]
        .mean()
        .rename(columns={"episode_steps": "mean_steps_per_episode"})
    )

    summary = mean_reward_by_agent.merge(mean_steps_by_agent, on="display_label", how="left")
    summary["training_time_minutes"] = summary["display_label"].map(TRAINING_TIME_MINUTES)
    summary["reward_per_minute"] = summary["mean_reward"] / summary["training_time_minutes"]
    summary["algorithm"] = summary["display_label"].map(algorithm_of)
    summary["representation"] = summary["display_label"].map(representation_of)
    order_map = {label: idx for idx, label in enumerate(DISPLAY_ORDER)}
    summary = summary.sort_values("display_label", key=lambda col: col.map(order_map)).reset_index(drop=True)
    summary.to_csv(efficiency_output_dir / "efficiency_summary.csv", index=False)

    table_df = summary[
        ["display_label", "mean_reward", "training_time_minutes", "mean_steps_per_episode", "reward_per_minute"]
    ].copy()
    table_df = table_df.rename(
        columns={
            "display_label": "Agent",
            "mean_reward": "Mean reward",
            "training_time_minutes": "Training time (min)",
            "mean_steps_per_episode": "Mean steps / episode",
            "reward_per_minute": "Reward / min",
        }
    )
    for column in ["Mean reward", "Training time (min)", "Mean steps / episode", "Reward / min"]:
        table_df[column] = table_df[column].map(lambda value: f"{float(value):.2f}")
    export_dataframe_table_figure(
        table_df,
        title="Efficiency summary",
        output_stem=efficiency_output_dir / "efficiency_summary_table",
        row_labels=summary["display_label"].tolist(),
    )

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for row in summary.itertuples(index=False):
        algo = row.algorithm
        rep = row.representation
        x = float(row.training_time_minutes)
        y = float(row.mean_reward)
        ax.scatter(x, y, s=85, color=ALGO_COLORS[algo], edgecolor="white", linewidth=0.9, zorder=4)
        ax.plot(
            [x - 0.7, x + 0.7],
            [y, y],
            color=ALGO_COLORS[algo],
            linestyle=REP_LINESTYLES[rep],
            linewidth=2.1,
            zorder=3,
        )
        ax.annotate(
            row.display_label,
            (x, y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10,
            color=ALGO_COLORS[algo],
        )

    ax.set_xlabel("Training time (minutes)")
    ax.set_ylabel("Mean reward")
    ax.set_title("Training efficiency: reward vs training time")
    algo_handles = [
        mlines.Line2D([0], [0], color=color, marker="o", linestyle="None", markersize=8, label=algo)
        for algo, color in ALGO_COLORS.items()
    ]
    rep_handles = [
        mlines.Line2D([0], [0], color="#444444", linestyle=linestyle, linewidth=2.2, label=rep)
        for rep, linestyle in REP_LINESTYLES.items()
    ]
    ax.legend(handles=algo_handles + rep_handles, ncol=2, frameon=True, loc="best")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(efficiency_output_dir / f"training_time_vs_reward.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return summary


def build_generalisation_summary(cross_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for algorithm in ["PPO", "DQN"]:
        for train_env in TRAIN_ENV_ORDER:
            candidate_short = short_label_for(train_env, algorithm)
            candidate_display = LABEL_MAP[candidate_short]
            for test_env in TEST_ENV_ORDER:
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
                    _, p_value = stats.wilcoxon(candidate_values, benchmark_values, zero_method="wilcox", alternative="two-sided")

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
                        "mean_reward": float(np.mean(candidate_values)),
                        "std_reward": float(np.std(candidate_values, ddof=1)) if len(candidate_values) > 1 else 0.0,
                        "mean_diff_vs_benchmark": float(np.mean(diffs)) if len(diffs) else 0.0,
                        "p_value_raw": float(p_value),
                        "n_seeds": int(len(paired)),
                    }
                )

    summary = pd.DataFrame(rows)
    adjusted_values = pd.Series(1.0, index=summary.index, dtype=float)
    for algorithm in ["PPO", "DQN"]:
        mask = (summary["algorithm"] == algorithm) & (
            summary["candidate_short_label"] != summary["benchmark_short_label"]
        )
        adjusted = holm_adjust(summary.loc[mask, "p_value_raw"].tolist())
        adjusted_values.loc[mask] = adjusted
    summary["p_value_holm"] = adjusted_values

    outcomes = []
    highlight_flags = []
    for row in summary.itertuples(index=False):
        if row.candidate_short_label == row.benchmark_short_label:
            outcome = "Draw"
        elif row.p_value_holm >= 0.05:
            outcome = "Draw"
        elif row.mean_diff_vs_benchmark > 0:
            outcome = "Win"
        elif row.mean_diff_vs_benchmark < 0:
            outcome = "Loss"
        else:
            outcome = "Draw"
        highlight = row.train_env == "masked" and outcome == "Win"
        outcomes.append(outcome)
        highlight_flags.append(highlight)
    summary["outcome"] = outcomes
    summary["highlight_masked_win"] = highlight_flags
    return summary


def generalisation_analysis(cross_seed_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    generalisation_output_dir = output_dir / "generalisation"
    generalisation_output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_generalisation_summary(cross_seed_df)
    summary.to_csv(generalisation_output_dir / "generalisation_longform.csv", index=False)

    for algorithm in ["PPO", "DQN"]:
        subset = summary[summary["algorithm"] == algorithm].copy()
        matrix_df = pd.DataFrame(index=[ENV_LABELS[env] for env in TRAIN_ENV_ORDER], columns=[ENV_LABELS[env] for env in TEST_ENV_ORDER])
        for row in subset.itertuples(index=False):
            cell_text = f"{mean_std_text(row.mean_reward, row.std_reward)}\n{row.outcome}"
            if row.highlight_masked_win:
                cell_text += " *"
            matrix_df.loc[row.train_env_label, row.test_env_label] = cell_text
        matrix_df.index.name = "Training environment"
        matrix_df = matrix_df.reset_index()
        matrix_df.to_csv(generalisation_output_dir / f"generalisation_matrix_{algorithm.lower()}.csv", index=False)

        heat_values = (
            subset.pivot(index="train_env_label", columns="test_env_label", values="mean_reward")
            .reindex(index=[ENV_LABELS[env] for env in TRAIN_ENV_ORDER], columns=[ENV_LABELS[env] for env in TEST_ENV_ORDER])
        )
        annotation_values = (
            subset.assign(
                annotation=subset.apply(
                    lambda frame_row: (
                        f"{mean_std_text(frame_row['mean_reward'], frame_row['std_reward'])}\n"
                        f"{frame_row['outcome']}{' *' if frame_row['highlight_masked_win'] else ''}"
                    ),
                    axis=1,
                )
            )
            .pivot(index="train_env_label", columns="test_env_label", values="annotation")
            .reindex(index=[ENV_LABELS[env] for env in TRAIN_ENV_ORDER], columns=[ENV_LABELS[env] for env in TEST_ENV_ORDER])
        )
        highlight_matrix = (
            subset.pivot(index="train_env_label", columns="test_env_label", values="highlight_masked_win")
            .reindex(index=[ENV_LABELS[env] for env in TRAIN_ENV_ORDER], columns=[ENV_LABELS[env] for env in TEST_ENV_ORDER])
            .fillna(False)
        )

        fig, ax = plt.subplots(figsize=(8.3, 5.4))
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
                    fontsize=9.5,
                    color="#111111",
                    bbox={
                        "boxstyle": "round,pad=0.26",
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

        note = "* masked-trained model outperforms the environment-matched benchmark"
        ax.text(0.0, -0.14, note, transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color="#2b8a3e")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(generalisation_output_dir / f"generalisation_matrix_{algorithm.lower()}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return summary


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    own_episode_df = load_own_environment_episode_data(input_dir)
    cross_seed_df = load_cross_environment_seed_data(input_dir)

    failure_summary, failure_threshold = failure_analysis(own_episode_df, output_dir)
    efficiency_summary = efficiency_analysis(own_episode_df, output_dir)
    generalisation_summary = generalisation_analysis(cross_seed_df, output_dir)

    manifest = pd.DataFrame(
        [
            {"analysis": "failure", "key_artifact": "failure/failure_summary.csv"},
            {"analysis": "failure", "key_artifact": "failure/episode_reward_distribution.png"},
            {"analysis": "failure", "key_artifact": "failure/reward_over_episode_index.png"},
            {"analysis": "efficiency", "key_artifact": "efficiency/efficiency_summary.csv"},
            {"analysis": "efficiency", "key_artifact": "efficiency/training_time_vs_reward.png"},
            {"analysis": "generalisation", "key_artifact": "generalisation/generalisation_longform.csv"},
            {"analysis": "generalisation", "key_artifact": "generalisation/generalisation_matrix_ppo.png"},
            {"analysis": "generalisation", "key_artifact": "generalisation/generalisation_matrix_dqn.png"},
        ]
    )
    manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)

    print(f"Saved outputs to: {output_dir}")
    print(f"PPO-RGB failure threshold (10th percentile): {failure_threshold:.3f}")
    print("\nFailure summary:")
    print(failure_summary[["display_label", "failed_episode_pct", "low_reward_pct"]].to_string(index=False))
    print("\nEfficiency summary:")
    print(efficiency_summary[["display_label", "mean_reward", "training_time_minutes", "mean_steps_per_episode", "reward_per_minute"]].to_string(index=False))
    print("\nGeneralisation summary:")
    print(generalisation_summary[["algorithm", "train_env_label", "test_env_label", "mean_reward", "std_reward", "outcome"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
