from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seeding import set_global_seed


OWN_SETTING_BY_MODEL = {
    "Best-RGB-optimized": "rgb",
    "Best-Masked-optimized": "masked",
    "Best-Greyscale-optimized": "grayscale",
    "Best-Grayscale-optimized": "grayscale",
}

PLOT_COLORS = [
    "#1f4e5f",
    "#39ff14",
    "#ff4fb3",
    "#8b5cf6",
    "#ff9f1c",
    "#ff6b6b",
]

PLOT_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
PLOT_MARKERS = ["o", "s", "^", "D", "v", "P"]


@dataclass(frozen=True)
class RunSpec:
    model_label: str
    agent: str
    setting: str
    checkpoint: Path
    eval_json: Path

    @property
    def run_id(self) -> str:
        return f"{self.model_label}/{self.agent}/{self.setting}"

    @property
    def short_label(self) -> str:
        return f"{self.model_label}/{self.agent}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisi statistica dei best trial con ranking e traiettorie.")
    parser.add_argument("--input-dir", default="models_checkpoints", help="Directory base contenente i checkpoint.")
    parser.add_argument("--output-dir", default="results/best_trial_analysis", help="Directory di output.")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--max-picked-episodes", type=int, default=3)
    parser.add_argument("--skip-trajectories", action="store_true", help="Salta la ricostruzione step-by-step.")
    return parser.parse_args()


def discover_best_runs(input_dir: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for model_label, own_setting in OWN_SETTING_BY_MODEL.items():
        model_dir = input_dir / model_label
        if not model_dir.exists():
            continue

        for agent_dir in sorted(model_dir.iterdir()):
            best_trial_dir = agent_dir / "best_trial"
            if not best_trial_dir.exists():
                continue

            agent = agent_dir.name.lower()
            eval_json = best_trial_dir / f"{agent}_model_eval_{own_setting}.json"
            checkpoint = best_trial_dir / f"{agent}_model.pth"
            if not eval_json.exists() or not checkpoint.exists():
                continue

            runs.append(
                RunSpec(
                    model_label=model_label,
                    agent=agent,
                    setting=own_setting,
                    checkpoint=checkpoint,
                    eval_json=eval_json,
                )
            )

    return sorted(runs, key=lambda run: (run.model_label, run.agent))


def load_run_data(runs: list[RunSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict] = []
    seed_rows: list[dict] = []

    for run in runs:
        with run.eval_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        per_seed = data.get("per_seed", [])
        if not per_seed:
            raise ValueError(f"Il file {run.eval_json} non contiene 'per_seed', richiesto per l'analisi.")

        for seed_block in per_seed:
            seed = int(seed_block["seed"])
            rewards = [float(value) for value in seed_block.get("episode_rewards", [])]

            seed_rows.append(
                {
                    "run_id": run.run_id,
                    "short_label": run.short_label,
                    "model_label": run.model_label,
                    "agent": run.agent,
                    "setting": run.setting,
                    "seed": seed,
                    "mean_reward_seed": float(np.mean(rewards)),
                    "std_reward_seed": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
                    "episodes_in_seed": len(rewards),
                }
            )

            for episode_index, reward in enumerate(rewards, start=1):
                episode_rows.append(
                    {
                        "run_id": run.run_id,
                        "short_label": run.short_label,
                        "model_label": run.model_label,
                        "agent": run.agent,
                        "setting": run.setting,
                        "seed": seed,
                        "episode_index": episode_index,
                        "reward": reward,
                    }
                )

    return pd.DataFrame(episode_rows), pd.DataFrame(seed_rows)


def add_confidence_interval(mean: float, sample_std: float, n: int, confidence_level: float) -> tuple[float, float]:
    if n <= 1:
        return mean, mean

    alpha = 1.0 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    se = sample_std / math.sqrt(n)
    margin = float(t_crit * se)
    return mean - margin, mean + margin


def build_ranking(seed_df: pd.DataFrame, episode_df: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    ranking = (
        seed_df.groupby(["run_id", "short_label", "model_label", "agent", "setting"], as_index=False)
        .agg(
            mean_seed_reward=("mean_reward_seed", "mean"),
            std_seed_reward=("mean_reward_seed", "std"),
            median_seed_reward=("mean_reward_seed", "median"),
            n_seeds=("seed", "nunique"),
        )
    )

    episode_summary = (
        episode_df.groupby("run_id", as_index=False)
        .agg(
            episode_mean_reward=("reward", "mean"),
            episode_std_reward=("reward", "std"),
            total_episodes=("reward", "size"),
        )
    )
    ranking = ranking.merge(episode_summary, on="run_id", how="left")

    ci_lows = []
    ci_highs = []
    for row in ranking.itertuples(index=False):
        ci_low, ci_high = add_confidence_interval(
            mean=float(row.mean_seed_reward),
            sample_std=float(0.0 if pd.isna(row.std_seed_reward) else row.std_seed_reward),
            n=int(row.n_seeds),
            confidence_level=confidence_level,
        )
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)

    ranking["ci_low"] = ci_lows
    ranking["ci_high"] = ci_highs
    ranking = ranking.sort_values(["mean_seed_reward", "episode_mean_reward"], ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def run_friedman(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = seed_df.pivot(index="seed", columns="run_id", values="mean_reward_seed").sort_index()
    pivot = pivot.dropna(axis=0, how="any").sort_index(axis=1)
    if pivot.empty or pivot.shape[1] < 3:
        raise ValueError("Servono almeno 3 run con seed allineati per il test di Friedman.")

    statistic, p_value = stats.friedmanchisquare(*[pivot[col].to_numpy() for col in pivot.columns])
    summary = pd.DataFrame(
        [
            {
                "test": "friedmanchisquare",
                "n_runs": pivot.shape[1],
                "n_seeds": pivot.shape[0],
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        ]
    )
    return pivot, summary


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        candidate = factor * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    non_zero = differences[differences != 0]
    if non_zero.size == 0:
        return 0.0
    abs_ranks = stats.rankdata(np.abs(non_zero))
    positive = float(abs_ranks[non_zero > 0].sum())
    negative = float(abs_ranks[non_zero < 0].sum())
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def build_pairwise_comparisons(seed_pivot: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    ordered_run_ids = ranking["run_id"].tolist()
    rows: list[dict] = []
    raw_p_values: list[float] = []

    for left_index, left_run in enumerate(ordered_run_ids):
        for right_run in ordered_run_ids[left_index + 1:]:
            paired = seed_pivot[[left_run, right_run]].dropna()
            left = paired[left_run].to_numpy()
            right = paired[right_run].to_numpy()
            differences = left - right

            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                statistic, p_value = stats.wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")

            row = {
                "run_a": left_run,
                "run_b": right_run,
                "n_seeds": int(len(paired)),
                "mean_run_a": float(np.mean(left)),
                "mean_run_b": float(np.mean(right)),
                "mean_diff_a_minus_b": float(np.mean(differences)),
                "median_diff_a_minus_b": float(np.median(differences)),
                "wilcoxon_statistic": float(statistic),
                "p_value_raw": float(p_value),
                "rank_biserial": float(rank_biserial_from_differences(differences)),
                "run_a_better_on_seed_count": int(np.sum(differences > 0)),
                "run_b_better_on_seed_count": int(np.sum(differences < 0)),
                "ties_on_seed_count": int(np.sum(differences == 0)),
            }
            rows.append(row)
            raw_p_values.append(float(p_value))

    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(rows, adjusted):
        row["p_value_holm"] = float(adjusted_p)
        row["significant_0_05"] = bool(adjusted_p < 0.05)

    return pd.DataFrame(rows).sort_values(["p_value_holm", "p_value_raw", "mean_diff_a_minus_b"], ascending=[True, True, False])


def pick_rank_consistent_episodes(episode_df: pd.DataFrame, ranking: pd.DataFrame, max_picks: int) -> pd.DataFrame:
    ordered_run_ids = ranking["run_id"].tolist()
    pivot = (
        episode_df.pivot_table(index=["seed", "episode_index"], columns="run_id", values="reward", aggfunc="first")
        .dropna()
        .reset_index()
    )

    exact_picks: list[dict] = []
    fallback_picks: list[dict] = []
    used_seeds: set[int] = set()

    for _, row in pivot.iterrows():
        values = np.array([float(row[run_id]) for run_id in ordered_run_ids], dtype=float)
        diffs = np.diff(values)
        strict_count = int(np.sum(diffs < 0))
        total_margin = float(np.sum(values[:-1] - values[1:]))
        candidate = {
            "seed": int(row["seed"]),
            "episode_index": int(row["episode_index"]),
            "strict_comparisons": strict_count,
            "total_margin": total_margin,
            "spearman_rho": float(stats.spearmanr(np.arange(len(values)), stats.rankdata(-values, method="average")).statistic),
            "respect_mode": "exact" if np.all(diffs <= 0) else "near",
        }
        for pos, run_id in enumerate(ordered_run_ids, start=1):
            candidate[f"rank_{pos}_run_id"] = run_id
            candidate[f"rank_{pos}_reward"] = float(values[pos - 1])

        if np.all(diffs <= 0):
            exact_picks.append(candidate)
        else:
            fallback_picks.append(candidate)

    exact_df = pd.DataFrame(exact_picks)
    fallback_df = pd.DataFrame(fallback_picks)

    if exact_df.empty and fallback_df.empty:
        return pd.DataFrame()

    selected_rows = []

    if not exact_df.empty:
        exact_df = exact_df.sort_values(
            ["strict_comparisons", "total_margin", "seed", "episode_index"],
            ascending=[False, False, True, True],
        )
        for row in exact_df.itertuples(index=False):
            if row.seed in used_seeds and len(selected_rows) < max_picks:
                continue
            selected_rows.append(row._asdict())
            used_seeds.add(row.seed)
            if len(selected_rows) >= max_picks:
                break

    if len(selected_rows) >= max_picks:
        return pd.DataFrame(selected_rows)

    if fallback_df.empty:
        return pd.DataFrame(selected_rows)

    fallback_df = fallback_df.sort_values(
        ["spearman_rho", "strict_comparisons", "total_margin", "seed", "episode_index"],
        ascending=[False, False, False, True, True],
    )
    existing = {(item["seed"], item["episode_index"]) for item in selected_rows}
    for row in fallback_df.itertuples(index=False):
        key = (row.seed, row.episode_index)
        if key in existing:
            continue
        if row.seed in used_seeds and len(selected_rows) < max_picks:
            continue
        selected_rows.append(row._asdict())
        used_seeds.add(row.seed)
        existing.add(key)
        if len(selected_rows) >= max_picks:
            break

    if len(selected_rows) >= max_picks:
        return pd.DataFrame(selected_rows)

    combined_df = pd.concat([exact_df, fallback_df], ignore_index=True).sort_values(
        ["strict_comparisons", "total_margin", "seed", "episode_index"],
        ascending=[False, False, True, True],
    )
    existing = {(item["seed"], item["episode_index"]) for item in selected_rows}
    for row in combined_df.itertuples(index=False):
        row_dict = row._asdict()
        key = (row_dict["seed"], row_dict["episode_index"])
        if key in existing:
            continue
        selected_rows.append(row_dict)
        existing.add(key)
        if len(selected_rows) >= max_picks:
            break

    return pd.DataFrame(selected_rows)


def compute_episode_alignment(episode_df: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    ordered_run_ids = ranking["run_id"].tolist()
    pivot = (
        episode_df.pivot_table(index=["seed", "episode_index"], columns="run_id", values="reward", aggfunc="first")
        .dropna()
        .reset_index()
    )

    rows: list[dict] = []
    for _, row in pivot.iterrows():
        values = np.array([float(row[run_id]) for run_id in ordered_run_ids], dtype=float)
        diffs = np.diff(values)
        alignment = {
            "seed": int(row["seed"]),
            "episode_index": int(row["episode_index"]),
            "strict_comparisons": int(np.sum(diffs < 0)),
            "total_margin": float(np.sum(values[:-1] - values[1:])),
            "spearman_rho": float(stats.spearmanr(np.arange(len(values)), stats.rankdata(-values, method="average")).statistic),
            "respect_mode": "exact" if np.all(diffs <= 0) else "near",
        }
        for pos, run_id in enumerate(ordered_run_ids, start=1):
            alignment[f"rank_{pos}_run_id"] = run_id
            alignment[f"rank_{pos}_reward"] = float(values[pos - 1])
        rows.append(alignment)

    return pd.DataFrame(rows)


def select_episode_regimes(alignment_df: pd.DataFrame, max_per_group: int = 3) -> pd.DataFrame:
    if alignment_df.empty:
        return pd.DataFrame()

    selected: list[dict] = []
    used_keys: set[tuple[int, int]] = set()

    def pick(df: pd.DataFrame, label: str) -> None:
        local_used_seeds: set[int] = set()
        for row in df.itertuples(index=False):
            key = (int(row.seed), int(row.episode_index))
            if key in used_keys:
                continue
            if row.seed in local_used_seeds and len(local_used_seeds) < max_per_group:
                continue
            item = row._asdict()
            item["regime"] = label
            selected.append(item)
            used_keys.add(key)
            local_used_seeds.add(int(row.seed))
            if sum(1 for x in selected if x["regime"] == label) >= max_per_group:
                break

        if sum(1 for x in selected if x["regime"] == label) >= max_per_group:
            return

        for row in df.itertuples(index=False):
            key = (int(row.seed), int(row.episode_index))
            if key in used_keys:
                continue
            item = row._asdict()
            item["regime"] = label
            selected.append(item)
            used_keys.add(key)
            if sum(1 for x in selected if x["regime"] == label) >= max_per_group:
                break

    optimistic_df = alignment_df.sort_values(
        ["spearman_rho", "strict_comparisons", "total_margin", "seed", "episode_index"],
        ascending=[False, False, False, True, True],
    )
    pick(optimistic_df, "optimistic")

    medium_pool = alignment_df[
        (alignment_df["spearman_rho"] >= 0.0)
        & (alignment_df["total_margin"] >= 0.0)
        & (alignment_df["strict_comparisons"] >= 2)
    ].copy()
    if medium_pool.empty:
        medium_pool = alignment_df.copy()

    med_rho = float(medium_pool["spearman_rho"].median())
    med_margin = float(medium_pool["total_margin"].median())
    med_strict = float(medium_pool["strict_comparisons"].median())
    medium_df = medium_pool.copy()
    medium_df["medium_score"] = (
        (medium_df["spearman_rho"] - med_rho).abs()
        + 0.15 * (medium_df["total_margin"] - med_margin).abs()
        + 0.25 * (medium_df["strict_comparisons"] - med_strict).abs()
    )
    medium_df = medium_df.sort_values(["medium_score", "seed", "episode_index"], ascending=[True, True, True])
    pick(medium_df, "medium")

    pessimistic_df = alignment_df.sort_values(
        ["spearman_rho", "strict_comparisons", "total_margin", "seed", "episode_index"],
        ascending=[True, True, True, True, True],
    )
    pick(pessimistic_df, "pessimistic")

    order = {"optimistic": 0, "medium": 1, "pessimistic": 2}
    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        return selected_df
    selected_df["regime_order"] = selected_df["regime"].map(order)
    selected_df = selected_df.sort_values(["regime_order", "seed", "episode_index"]).drop(columns=["regime_order"])
    return selected_df.reset_index(drop=True)


def import_atari_dependencies() -> None:
    import ale_py  # noqa: F401
    import gymnasium as gym

    gym.register_envs(ale_py)


def build_setting_kwargs(setting: str) -> tuple[dict, dict]:
    common_cfg = load_config(PROJECT_ROOT / "configs/config_common.yaml")
    preprocessing_kwargs = common_cfg.preprocessing.to_dict()
    masking_kwargs = common_cfg.masking.to_dict()

    if setting == "rgb":
        preprocessing_kwargs["mode"] = "rgb"
        masking_kwargs["enabled"] = False
    elif setting == "masked":
        preprocessing_kwargs["mode"] = "rgb"
        masking_kwargs["enabled"] = True
    elif setting == "grayscale":
        preprocessing_kwargs["mode"] = "grayscale"
        masking_kwargs["enabled"] = False
    else:
        raise ValueError(f"Setting non supportato: {setting}")

    return preprocessing_kwargs, masking_kwargs


def load_agent(run: RunSpec):
    import torch

    from src.agents.dqn_agent import DQNAgent
    from src.agents.ppo_agent import PPOAgent
    from src.envs.atari_wrappers import make_atari_env

    common_cfg = load_config(PROJECT_ROOT / "configs/config_common.yaml")
    preprocessing_kwargs, masking_kwargs = build_setting_kwargs(run.setting)
    probe_env = make_atari_env(
        common_cfg.env.id,
        seed=common_cfg.seed + 10000,
        preprocessing_kwargs=preprocessing_kwargs,
        masking_kwargs=masking_kwargs,
    )
    obs_shape = probe_env.observation_space.shape
    n_actions = probe_env.action_space.n
    probe_env.close()

    if run.agent == "dqn":
        cfg_dqn = load_config(PROJECT_ROOT / "configs/config_dqn.yaml")
        agent = DQNAgent(
            n_actions=n_actions,
            feature_dim=cfg_dqn.model.feature_dim,
            learning_rate=cfg_dqn.dqn.learning_rate,
            gamma=cfg_dqn.dqn.gamma,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            device="cpu",
            in_channels=obs_shape[0],
        )
        agent.load(str(run.checkpoint))
        agent.set_training_mode(False)
        return agent

    cfg_ppo = load_config(PROJECT_ROOT / "configs/config_ppo.yaml")
    agent = PPOAgent(
        n_actions=n_actions,
        n_envs=1,
        n_steps=cfg_ppo.ppo.n_steps,
        n_epochs=cfg_ppo.ppo.n_epochs,
        batch_size=cfg_ppo.ppo.batch_size,
        learning_rate=cfg_ppo.ppo.learning_rate,
        clip_range=cfg_ppo.ppo.clip_range,
        gamma=cfg_ppo.ppo.gamma,
        gae_lambda=cfg_ppo.ppo.gae_lambda,
        ent_coef=cfg_ppo.ppo.ent_coef,
        vf_coef=cfg_ppo.ppo.vf_coef,
        clip_range_vf=getattr(cfg_ppo.ppo, "clip_range_vf", None),
        normalize_advantage=getattr(cfg_ppo.ppo, "normalize_advantage", True),
        max_grad_norm=getattr(cfg_ppo.ppo, "max_grad_norm", 0.5),
        device="cpu",
        feature_dim=getattr(cfg_ppo.model, "feature_dim", 512),
        obs_shape=obs_shape,
    )
    agent.load(str(run.checkpoint))
    agent.net.eval()
    return agent


def replay_episode_trajectory(run: RunSpec, seed: int, episode_index: int) -> pd.DataFrame:
    import_atari_dependencies()
    from src.envs.atari_wrappers import make_atari_env
    from src.envs.vec_env import make_eval_vec_env

    common_cfg = load_config(PROJECT_ROOT / "configs/config_common.yaml")
    preprocessing_kwargs, masking_kwargs = build_setting_kwargs(run.setting)
    agent = load_agent(run)
    rows: list[dict] = []

    if run.agent == "ppo":
        set_global_seed(seed)
        env = make_eval_vec_env(
            common_cfg.env.id,
            seed=seed,
            preprocessing_kwargs=preprocessing_kwargs,
            masking_kwargs=masking_kwargs,
        )
        try:
            obs = env.reset()
            for current_episode in range(1, episode_index + 1):
                done = False
                step_idx = 0
                cumulative_reward = 0.0

                while not done:
                    action = agent.act(np.asarray(obs)[0])
                    obs, reward, done_arr, _ = env.step(np.array([action]))
                    clipped_reward = float(reward[0])
                    step_idx += 1
                    cumulative_reward += clipped_reward
                    done = bool(done_arr[0])

                    if current_episode == episode_index:
                        rows.append(
                            {
                                "run_id": run.run_id,
                                "short_label": run.short_label,
                                "seed": seed,
                                "episode_index": episode_index,
                                "step": step_idx,
                                "reward": clipped_reward,
                                "cumulative_reward": cumulative_reward,
                            }
                        )
        finally:
            env.close()
    else:
        set_global_seed(seed)
        env = make_atari_env(
            common_cfg.env.id,
            seed=seed,
            preprocessing_kwargs=preprocessing_kwargs,
            masking_kwargs=masking_kwargs,
        )
        try:
            for current_episode in range(1, episode_index + 1):
                obs, _ = env.reset()
                done = False
                step_idx = 0
                cumulative_reward = 0.0

                while not done:
                    action = agent.act(np.asarray(obs))
                    obs, reward, terminated, truncated, _ = env.step(action)
                    clipped_reward = float(reward)
                    step_idx += 1
                    cumulative_reward += clipped_reward
                    done = bool(terminated or truncated)

                    if current_episode == episode_index:
                        rows.append(
                            {
                                "run_id": run.run_id,
                                "short_label": run.short_label,
                                "seed": seed,
                                "episode_index": episode_index,
                                "step": step_idx,
                                "reward": clipped_reward,
                                "cumulative_reward": cumulative_reward,
                            }
                        )
        finally:
            env.close()

    return pd.DataFrame(rows)


def save_temporal_plots(
    selected_df: pd.DataFrame,
    runs: list[RunSpec],
    ranking_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    run_lookup = {run.run_id: run for run in runs}
    ranking_order = ranking_df["run_id"].tolist()
    trajectories: list[pd.DataFrame] = []
    saved_paths: list[Path] = []

    for selected in selected_df.itertuples(index=False):
        episode_frames: list[pd.DataFrame] = []
        for run_id in ranking_order:
            frame = replay_episode_trajectory(
                run=run_lookup[run_id],
                seed=int(selected.seed),
                episode_index=int(selected.episode_index),
            )
            episode_frames.append(frame)

        combined = pd.concat(episode_frames, ignore_index=True)
        combined["picked_seed"] = int(selected.seed)
        combined["picked_episode"] = int(selected.episode_index)
        if hasattr(selected, "regime"):
            combined["regime"] = str(selected.regime)
        trajectories.append(combined)

        fig, ax = plt.subplots(figsize=(11, 6))
        for idx, (color, run_id) in enumerate(zip(PLOT_COLORS, ranking_order)):
            subset = combined[combined["run_id"] == run_id]
            ax.plot(
                subset["step"],
                subset["cumulative_reward"],
                label=subset["short_label"].iloc[0],
                color=color,
                linewidth=2.2,
                linestyle=PLOT_LINESTYLES[idx % len(PLOT_LINESTYLES)],
                marker=PLOT_MARKERS[idx % len(PLOT_MARKERS)],
                markersize=4.2,
                markevery=max(1, len(subset) // 10),
                alpha=0.95,
                zorder=10 + idx,
                path_effects=[pe.Stroke(linewidth=4.4, foreground="white"), pe.Normal()],
            )

        title_prefix = f"{selected.regime} | " if hasattr(selected, "regime") else ""
        ax.set_title(f"{title_prefix}Cumulative reward over time | seed={selected.seed} episode={selected.episode_index}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative clipped reward")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        if hasattr(selected, "regime"):
            path = plots_dir / f"temporal_{selected.regime}_seed_{selected.seed}_episode_{selected.episode_index}.png"
        else:
            path = plots_dir / f"temporal_seed_{selected.seed}_episode_{selected.episode_index}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        saved_paths.append(path)

    if not trajectories:
        return pd.DataFrame(), []

    trajectory_df = pd.concat(trajectories, ignore_index=True)
    return trajectory_df, saved_paths


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))

    runs = discover_best_runs(input_dir)
    if not runs:
        raise FileNotFoundError(f"Nessun best trial trovato in {input_dir}")

    episode_df, seed_df = load_run_data(runs)
    ranking_df = build_ranking(seed_df, episode_df, args.confidence_level)
    seed_pivot, friedman_df = run_friedman(seed_df)
    pairwise_df = build_pairwise_comparisons(seed_pivot, ranking_df)
    selected_df = pick_rank_consistent_episodes(episode_df, ranking_df, args.max_picked_episodes)
    alignment_df = compute_episode_alignment(episode_df, ranking_df)
    regime_df = select_episode_regimes(alignment_df, max_per_group=3)

    episode_df.to_csv(output_dir / "episode_level_results.csv", index=False)
    seed_df.to_csv(output_dir / "seed_level_results.csv", index=False)
    ranking_df.to_csv(output_dir / "ranking.csv", index=False)
    friedman_df.to_csv(output_dir / "omnibus_friedman.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_wilcoxon.csv", index=False)
    selected_df.to_csv(output_dir / "selected_rank_consistent_episodes.csv", index=False)
    alignment_df.to_csv(output_dir / "episode_alignment_scores.csv", index=False)
    regime_df.to_csv(output_dir / "selected_episode_regimes.csv", index=False)

    trajectory_paths: list[Path] = []
    if not args.skip_trajectories and not selected_df.empty:
        try:
            legacy_trajectory_df, legacy_paths = save_temporal_plots(selected_df, runs, ranking_df, output_dir)
            legacy_trajectory_df.to_csv(output_dir / "selected_episode_trajectories.csv", index=False)
            trajectory_paths.extend(legacy_paths)
            if not regime_df.empty:
                regime_trajectory_df, regime_paths = save_temporal_plots(regime_df, runs, ranking_df, output_dir)
                regime_trajectory_df.to_csv(output_dir / "selected_episode_regime_trajectories.csv", index=False)
                trajectory_paths.extend(regime_paths)
        except ModuleNotFoundError as exc:
            print(f"\nTraiettorie step-by-step saltate: dipendenza mancante ({exc}).")
        except Exception as exc:
            print(f"\nTraiettorie step-by-step non generate: {exc}")

    print("\nClassifica best trial (ordinata per mean reward sui mean-per-seed):")
    for row in ranking_df.itertuples(index=False):
        print(
            f"{row.rank}. {row.short_label} | setting={row.setting} "
            f"| mean_seed_reward={row.mean_seed_reward:.3f} "
            f"| CI95=[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        )

    print("\nTest omnibus:")
    omnibus = friedman_df.iloc[0]
    print(
        f"Friedman chi-square={omnibus['statistic']:.4f}, "
        f"p-value={omnibus['p_value']:.6f}, "
        f"runs={int(omnibus['n_runs'])}, seeds={int(omnibus['n_seeds'])}"
    )

    print(f"\nOutput salvati in: {output_dir}")
    print("CSV principali:")
    print(f"  - {output_dir / 'ranking.csv'}")
    print(f"  - {output_dir / 'pairwise_wilcoxon.csv'}")
    print(f"  - {output_dir / 'selected_rank_consistent_episodes.csv'}")
    if trajectory_paths:
        print("Grafici temporali:")
        for path in trajectory_paths:
            print(f"  - {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
