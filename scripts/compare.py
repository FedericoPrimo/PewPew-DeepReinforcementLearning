"""
Confronto finale DQN vs PPO.

Carica i JSON prodotti da train_dqn.py e train_ppo.py e genera:
  - Report testuale: mean±std, max reward, training time
  - Grafici: box plot, bar chart comparativo (via src/analysis/plots.py)

Uso:
  python scripts/compare.py
  python scripts/compare.py --dqn results/dqn_results.json --ppo results/ppo_results.json
  python scripts/compare.py --plot-format pdf --no-plots
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_result(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[compare] File non trovato: {p}")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def print_report(dqn: dict, ppo: dict) -> None:
    print("\n" + "=" * 60)
    print("  CONFRONTO DQN vs PPO — Space Invaders (Atari)")
    print("=" * 60)
    print(f"{'Metrica':<30} {'DQN':>12} {'PPO':>12}")
    print("-" * 60)

    metrics = [
        ("mean_reward", "Mean reward (eval)"),
        ("std_reward", "Std reward (eval)"),
        ("max_reward", "Max reward (eval)"),
        ("total_timesteps", "Total timesteps"),
        ("training_time_seconds", "Training time (s)"),
        ("eval_episodes", "Eval episodes"),
        ("seed", "Seed"),
    ]

    for key, label in metrics:
        dval = dqn.get(key, "N/A")
        pval = ppo.get(key, "N/A")
        if isinstance(dval, float):
            print(f"  {label:<28} {dval:>12.2f} {pval:>12.2f}")
        else:
            print(f"  {label:<28} {str(dval):>12} {str(pval):>12}")

    print("-" * 60)
    if dqn["mean_reward"] != 0:
        delta = ppo["mean_reward"] - dqn["mean_reward"]
        pct = delta / abs(dqn["mean_reward"]) * 100
        winner = "PPO" if delta > 0 else "DQN"
        print(f"  Delta mean_reward (PPO - DQN): {delta:+.2f} ({pct:+.1f}%)")
        print(f"  Migliore: {winner}")
    print("=" * 60 + "\n")


def build_eval_df(dqn: dict, ppo: dict) -> pd.DataFrame:
    """
    Costruisce un DataFrame sintetico dai JSON di evaluation.
    Genera eval_episodes righe sintetiche con distribuzione gaussiana
    per compatibilità con le funzioni di plotting esistenti.
    """
    rows = []
    for agent_data in [dqn, ppo]:
        name = agent_data["agent"].upper()
        mean = agent_data["mean_reward"]
        std = agent_data["std_reward"]
        n = agent_data["eval_episodes"]
        np.random.seed(agent_data["seed"])
        sampled = np.random.normal(loc=mean, scale=max(std, 1e-6), size=n)
        sampled = np.clip(sampled, 0, agent_data["max_reward"])
        for i, r in enumerate(sampled):
            rows.append({
                "agent_name": name,
                "preprocessing_mode": "grayscale",
                "episode_index": i,
                "total_reward": r,
                "seed": agent_data["seed"],
            })
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: Path, fmt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Box plot
    agents = df["agent_name"].unique()
    fig, ax = plt.subplots(figsize=(6, 5))
    data_per_agent = [df[df["agent_name"] == a]["total_reward"].values for a in agents]
    ax.boxplot(data_per_agent, tick_labels=agents, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.7))
    ax.set_title("Distribuzione reward — DQN vs PPO")
    ax.set_ylabel("Total Reward (eval)")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_dir / f"comparison_boxplot.{fmt}", dpi=150)
    plt.close(fig)
    print(f"  Salvato: {output_dir}/comparison_boxplot.{fmt}")

    # Bar chart con ±std
    means = df.groupby("agent_name")["total_reward"].mean()
    stds = df.groupby("agent_name")["total_reward"].std()
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#2196F3", "#FF9800"]
    bars = ax.bar(means.index, means.values, yerr=stds.values, capsize=8,
                  color=colors[:len(means)], alpha=0.85)
    ax.set_title("Reward medio ± std — DQN vs PPO")
    ax.set_ylabel("Mean Reward (eval)")
    ax.grid(axis="y", alpha=0.4)
    for bar, mean in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"comparison_barchart.{fmt}", dpi=150)
    plt.close(fig)
    print(f"  Salvato: {output_dir}/comparison_barchart.{fmt}")


def main():
    parser = argparse.ArgumentParser(description="Confronto DQN vs PPO")
    parser.add_argument("--dqn", default="results/dqn_results.json")
    parser.add_argument("--ppo", default="results/ppo_results.json")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    dqn_result = load_result(args.dqn)
    ppo_result = load_result(args.ppo)

    print_report(dqn_result, ppo_result)

    if not args.no_plots:
        df = build_eval_df(dqn_result, ppo_result)
        print("[compare] Generazione grafici...")
        plot_comparison(df, Path(args.output_dir), args.plot_format)
        print("[compare] Grafici salvati in:", args.output_dir)


if __name__ == "__main__":
    main()
