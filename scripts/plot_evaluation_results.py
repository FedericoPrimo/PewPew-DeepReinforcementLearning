"""
Genera grafici dai JSON prodotti da scripts/evaluate_model.py.

Esempi:
  python scripts/plot_evaluation_results.py
  python scripts/plot_evaluation_results.py --input-dir models_checkpoints --output-dir results/eval_plots
  python scripts/plot_evaluation_results.py --input-dir models_checkpoints --output-dir results/eval_plots_own --own-setting-only
  python scripts/plot_evaluation_results.py --input-dir models_checkpoints/Best-Masked-optimized --plot-format pdf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETTING_ORDER = ["rgb", "masked", "grayscale"]
SETTING_COLORS = {
    "rgb": "#2f80ed",
    "masked": "#f2994a",
    "grayscale": "#828282",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grafici per evaluation multiseed.")
    parser.add_argument("--input-dir", default="models_checkpoints", help="Directory in cui cercare *_eval_*.json.")
    parser.add_argument("--output-dir", default="results/eval_plots", help="Directory output per grafici e CSV.")
    parser.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument(
        "--own-setting-only",
        action="store_true",
        help="Usa solo RGB->rgb, Masked->masked, Greyscale/Grayscale->grayscale.",
    )
    return parser.parse_args()


def discover_eval_jsons(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*_eval_*.json"))


def checkpoint_label(checkpoint: str, json_path: Path) -> str:
    parts = Path(checkpoint).parts if checkpoint else json_path.parts
    for part in parts:
        if part.lower().startswith("best-"):
            return part

    text = str(json_path)
    match = re.search(r"(Best-[^\\/]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return json_path.parent.name


def load_eval_data(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows = []
    summary_rows = []

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        setting = data.get("setting", infer_setting_from_name(path))
        agent = data.get("agent", "unknown")
        label = checkpoint_label(data.get("checkpoint", ""), path)
        display_label = f"{label}/{agent}"
        run_id = f"{label}/{agent}/{setting}"

        per_seed = data.get("per_seed")
        if per_seed:
            for seed_data in per_seed:
                seed = int(seed_data["seed"])
                for episode_index, reward in enumerate(seed_data.get("episode_rewards", []), start=1):
                    episode_rows.append(
                        {
                            "run_id": run_id,
                            "model_label": label,
                            "display_label": display_label,
                            "agent": agent,
                            "setting": setting,
                            "seed": seed,
                            "episode_index": episode_index,
                            "reward": float(reward),
                            "json_path": str(path),
                        }
                    )
        else:
            seed = int(data.get("seed", 0))
            for episode_index, reward in enumerate(data.get("episode_rewards", []), start=1):
                episode_rows.append(
                    {
                        "run_id": run_id,
                        "model_label": label,
                        "display_label": display_label,
                        "agent": agent,
                        "setting": setting,
                        "seed": seed,
                        "episode_index": episode_index,
                        "reward": float(reward),
                        "json_path": str(path),
                    }
                )

        summary_rows.append(
            {
                "run_id": run_id,
                "model_label": label,
                "display_label": display_label,
                "agent": agent,
                "setting": setting,
                "mean_reward": float(data.get("mean_reward", np.nan)),
                "std_reward": float(data.get("std_reward", np.nan)),
                "min_reward": float(data.get("min_reward", np.nan)),
                "max_reward": float(data.get("max_reward", np.nan)),
                "num_seeds": int(data.get("num_seeds", 1)),
                "total_episodes": int(data.get("total_episodes", len(data.get("episode_rewards", [])))),
                "json_path": str(path),
            }
        )

    return pd.DataFrame(episode_rows), pd.DataFrame(summary_rows)


def aggregate_summary_for_plots(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collassa eventuali JSON duplicati per stesso modello/agente/setting.
    Puo' capitare se nella cartella ci sono piu' evaluation dello stesso checkpoint.
    """
    group_cols = ["display_label", "setting"]
    return (
        summary_df.groupby(group_cols, as_index=False)
        .agg(
            model_label=("model_label", "first"),
            agent=("agent", "first"),
            mean_reward=("mean_reward", "mean"),
            std_reward=("std_reward", "mean"),
            min_reward=("min_reward", "min"),
            max_reward=("max_reward", "max"),
            num_seeds=("num_seeds", "sum"),
            total_episodes=("total_episodes", "sum"),
        )
        .sort_values(["setting", "mean_reward"], ascending=[True, False])
    )


def expected_setting_for_model(model_label: str) -> str | None:
    label = model_label.lower()
    if "rgb" in label:
        return "rgb"
    if "masked" in label:
        return "masked"
    if "greyscale" in label or "grayscale" in label:
        return "grayscale"
    return None


def filter_own_setting(episodes_df: pd.DataFrame, summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_mask = summary_df.apply(
        lambda row: row["setting"] == expected_setting_for_model(str(row["model_label"])),
        axis=1,
    )
    episodes_mask = episodes_df.apply(
        lambda row: row["setting"] == expected_setting_for_model(str(row["model_label"])),
        axis=1,
    )
    return episodes_df[episodes_mask].copy(), summary_df[summary_mask].copy()


def infer_setting_from_name(path: Path) -> str:
    name = path.stem.lower()
    for setting in SETTING_ORDER:
        if setting in name:
            return setting
    return "unknown"


def ordered_settings(values: pd.Series) -> list[str]:
    present = list(dict.fromkeys(values.dropna().tolist()))
    ordered = [setting for setting in SETTING_ORDER if setting in present]
    ordered.extend(setting for setting in present if setting not in ordered)
    return ordered


def save_summary_tables(episodes_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.sort_values(["setting", "mean_reward"], ascending=[True, False]).to_csv(
        output_dir / "evaluation_summary.csv",
        index=False,
    )

    seed_df = (
        episodes_df.groupby(["display_label", "model_label", "agent", "setting", "seed"], as_index=False)
        .agg(mean_reward=("reward", "mean"), std_reward=("reward", "std"), episodes=("reward", "count"))
        .sort_values(["setting", "model_label", "seed"])
    )
    seed_df.to_csv(output_dir / "evaluation_by_seed.csv", index=False)


def plot_grouped_bars(summary_df: pd.DataFrame, output_dir: Path, fmt: str) -> Path:
    settings = ordered_settings(summary_df["setting"])
    models = sorted(summary_df["display_label"].unique())
    x = np.arange(len(models))
    width = 0.8 / max(len(settings), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 5.5))
    for idx, setting in enumerate(settings):
        subset = summary_df[summary_df["setting"] == setting].set_index("display_label")
        means = [subset.loc[model, "mean_reward"] if model in subset.index else np.nan for model in models]
        stds = [subset.loc[model, "std_reward"] if model in subset.index else np.nan for model in models]
        offset = (idx - (len(settings) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=setting,
            color=SETTING_COLORS.get(setting),
            alpha=0.85,
        )

    ax.set_title("Mean reward by model and evaluation setting")
    ax.set_ylabel("Reward")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Eval setting")
    fig.tight_layout()

    path = output_dir / f"mean_reward_by_model_setting.{fmt}"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_heatmap(summary_df: pd.DataFrame, output_dir: Path, fmt: str) -> Path:
    pivot = summary_df.pivot_table(
        index="display_label",
        columns="setting",
        values="mean_reward",
        aggfunc="mean",
    )
    cols = [setting for setting in SETTING_ORDER if setting in pivot.columns]
    cols.extend(col for col in pivot.columns if col not in cols)
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.6), max(4, len(pivot) * 0.55)))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_title("Mean reward heatmap")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, label="Mean reward")
    fig.tight_layout()

    path = output_dir / f"mean_reward_heatmap.{fmt}"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_episode_boxplots(episodes_df: pd.DataFrame, output_dir: Path, fmt: str) -> list[Path]:
    paths = []
    for setting in ordered_settings(episodes_df["setting"]):
        subset = episodes_df[episodes_df["setting"] == setting]
        models = sorted(subset["display_label"].unique())
        data = [subset[subset["display_label"] == model]["reward"].values for model in models]

        fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 5.2))
        ax.boxplot(data, labels=models, patch_artist=True)
        ax.set_title(f"Episode reward distribution - {setting}")
        ax.set_ylabel("Reward")
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"episode_reward_boxplot_{setting}.{fmt}"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_seed_means(episodes_df: pd.DataFrame, output_dir: Path, fmt: str) -> Path:
    seed_df = episodes_df.groupby(["display_label", "setting", "seed"], as_index=False)["reward"].mean()
    settings = ordered_settings(seed_df["setting"])
    models = sorted(seed_df["display_label"].unique())

    fig, axes = plt.subplots(len(settings), 1, figsize=(max(8, len(models) * 1.2), 3.2 * len(settings)), sharex=False)
    if len(settings) == 1:
        axes = [axes]

    for ax, setting in zip(axes, settings):
        subset = seed_df[seed_df["setting"] == setting]
        for model in models:
            model_df = subset[subset["display_label"] == model].sort_values("seed")
            if model_df.empty:
                continue
            ax.plot(model_df["seed"], model_df["reward"], marker="o", linewidth=1.6, label=model)
        ax.set_title(f"Mean reward per seed - {setting}")
        ax.set_ylabel("Reward")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Seed")
    fig.tight_layout()

    path = output_dir / f"mean_reward_per_seed.{fmt}"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_seed_means_single_chart(episodes_df: pd.DataFrame, output_dir: Path, fmt: str) -> Path:
    seed_df = (
        episodes_df.groupby(["display_label", "setting", "seed"], as_index=False)["reward"]
        .mean()
        .sort_values(["display_label", "setting", "seed"])
    )
    seed_df["series_label"] = seed_df["display_label"] + " [" + seed_df["setting"] + "]"
    series = sorted(seed_df["series_label"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(series) * 0.9), 6.2))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for idx, label in enumerate(series):
        data = seed_df[seed_df["series_label"] == label].sort_values("seed")
        setting = str(data["setting"].iloc[0])
        ax.plot(
            data["seed"],
            data["reward"],
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=5,
            label=label,
            color=SETTING_COLORS.get(setting),
            alpha=0.9,
        )

    ax.set_title("Mean reward per seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean reward")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    path = output_dir / f"mean_reward_per_seed_single_chart.{fmt}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_episode_mean_over_time(episodes_df: pd.DataFrame, output_dir: Path, fmt: str) -> Path:
    episode_df = (
        episodes_df.groupby(["display_label", "setting", "episode_index"], as_index=False)
        .agg(mean_reward=("reward", "mean"), std_reward=("reward", "std"))
        .sort_values(["display_label", "setting", "episode_index"])
    )
    episode_df["series_label"] = episode_df["display_label"] + " [" + episode_df["setting"] + "]"
    series = sorted(episode_df["series_label"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(series) * 0.9), 6.2))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for idx, label in enumerate(series):
        data = episode_df[episode_df["series_label"] == label].sort_values("episode_index")
        setting = str(data["setting"].iloc[0])
        color = SETTING_COLORS.get(setting)
        x = data["episode_index"].to_numpy(dtype=float)
        y = data["mean_reward"].to_numpy(dtype=float)
        std = data["std_reward"].fillna(0.0).to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=4,
            label=label,
            color=color,
            alpha=0.95,
        )
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.08)

    ax.set_title("Mean reward per episode index")
    ax.set_xlabel("Episode index within each seed")
    ax.set_ylabel("Mean reward across seeds")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    path = output_dir / f"mean_reward_per_episode_over_time.{fmt}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_episode_rolling_timeline(episodes_df: pd.DataFrame, output_dir: Path, fmt: str) -> Path:
    timeline = episodes_df.sort_values(["display_label", "setting", "seed", "episode_index"]).copy()
    timeline["series_label"] = timeline["display_label"] + " [" + timeline["setting"] + "]"
    timeline["global_episode"] = timeline.groupby("series_label").cumcount() + 1
    timeline["rolling_reward"] = timeline.groupby("series_label")["reward"].transform(
        lambda values: values.rolling(window=10, min_periods=1).mean()
    )
    series = sorted(timeline["series_label"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(series) * 0.9), 6.2))
    for label in series:
        data = timeline[timeline["series_label"] == label]
        setting = str(data["setting"].iloc[0])
        ax.plot(
            data["global_episode"],
            data["rolling_reward"],
            linewidth=1.8,
            label=label,
            color=SETTING_COLORS.get(setting),
            alpha=0.9,
        )

    ax.set_title("Rolling mean reward over evaluation episodes")
    ax.set_xlabel("Evaluation episode sequence")
    ax.set_ylabel("Rolling mean reward, window=10")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    path = output_dir / f"rolling_reward_episode_timeline.{fmt}"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def print_best_by_setting(summary_df: pd.DataFrame) -> None:
    print("\nBest model by evaluation setting")
    print("=" * 70)
    for setting in ordered_settings(summary_df["setting"]):
        subset = summary_df[summary_df["setting"] == setting].sort_values("mean_reward", ascending=False)
        if subset.empty:
            continue
        best = subset.iloc[0]
        print(
            f"{setting:<10} {best['display_label']:<30} "
            f"mean={best['mean_reward']:.2f} std={best['std_reward']:.2f} "
            f"episodes={best['total_episodes']}"
        )


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    paths = discover_eval_jsons(input_dir)

    if not paths:
        raise FileNotFoundError(f"Nessun file *_eval_*.json trovato in {input_dir}")

    episodes_df, summary_df = load_eval_data(paths)
    if episodes_df.empty or summary_df.empty:
        raise ValueError("I JSON trovati non contengono episode_rewards validi.")

    if args.own_setting_only:
        episodes_df, summary_df = filter_own_setting(episodes_df, summary_df)
        if episodes_df.empty or summary_df.empty:
            raise ValueError("Nessun JSON corrisponde al proprio setting di training.")

    plot_summary_df = aggregate_summary_for_plots(summary_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_summary_tables(episodes_df, summary_df, output_dir)

    plot_paths = [
        plot_grouped_bars(plot_summary_df, output_dir, args.plot_format),
        plot_heatmap(plot_summary_df, output_dir, args.plot_format),
        plot_seed_means(episodes_df, output_dir, args.plot_format),
        plot_seed_means_single_chart(episodes_df, output_dir, args.plot_format),
        plot_episode_mean_over_time(episodes_df, output_dir, args.plot_format),
        plot_episode_rolling_timeline(episodes_df, output_dir, args.plot_format),
    ]
    plot_paths.extend(plot_episode_boxplots(episodes_df, output_dir, args.plot_format))

    print_best_by_setting(plot_summary_df)
    print(f"\nGrafici salvati in: {output_dir}")
    for path in plot_paths:
        print(f"  {path}")
    print(f"\nTabelle salvate:")
    print(f"  {output_dir / 'evaluation_summary.csv'}")
    print(f"  {output_dir / 'evaluation_by_seed.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
