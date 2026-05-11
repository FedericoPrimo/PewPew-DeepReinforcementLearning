"""
Grafici delle performance degli agenti.
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_reward_distribution(
    df: pd.DataFrame,
    output_dir: str | Path = "results/plots",
    format: str = "png",
    group_col: str = "agent_name",
) -> Path:
    """
    Box plot della distribuzione dei reward per agente/modalità.

    Args:
        df: DataFrame con colonne `total_reward` e `group_col`.
        output_dir: Directory di output.
        format: Formato immagine ("png", "pdf", "svg").
        group_col: Colonna per il raggruppamento.

    Returns:
        Percorso al file salvato.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    groups = df[group_col].unique()
    data_per_group = [df[df[group_col] == g]["total_reward"].values for g in groups]

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 2), 5))
    ax.boxplot(data_per_group, labels=groups, patch_artist=True)
    ax.set_title("Distribuzione reward per agente")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Total Reward")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    filepath = output_dir / f"reward_distribution.{format}"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Salvato: {filepath}")
    return filepath


def plot_reward_over_episodes(
    df: pd.DataFrame,
    output_dir: str | Path = "results/plots",
    format: str = "png",
    hue_col: str = "preprocessing_mode",
) -> Path:
    """
    Reward per episodio, separato per modalità di preprocessing.

    Args:
        df: DataFrame con `episode_index`, `total_reward`, `hue_col`.
        output_dir: Directory di output.
        format: Formato immagine.
        hue_col: Colonna per separare le linee.

    Returns:
        Percorso al file salvato.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df[hue_col].unique()

    for group in groups:
        subset = df[df[hue_col] == group]
        avg = subset.groupby("episode_index")["total_reward"].mean()
        ax.plot(avg.index, avg.values, label=str(group), marker="o", markersize=3)

    ax.set_title("Reward medio per episodio")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Reward medio")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()

    filepath = output_dir / f"reward_over_episodes.{format}"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Salvato: {filepath}")
    return filepath


def plot_mean_reward_comparison(
    df: pd.DataFrame,
    output_dir: str | Path = "results/plots",
    format: str = "png",
    group_cols: List[str] = None,
) -> Path:
    """
    Bar chart del reward medio con intervalli di confidenza al 95%.

    Args:
        df: DataFrame con i risultati.
        output_dir: Directory di output.
        format: Formato immagine.
        group_cols: Colonne per il raggruppamento.

    Returns:
        Percorso al file salvato.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    group_cols = group_cols or ["agent_name", "preprocessing_mode"]

    # Crea etichetta combinata
    df = df.copy()
    df["_group"] = df[group_cols].astype(str).agg(" | ".join, axis=1)

    grouped = df.groupby("_group")["total_reward"]
    means = grouped.mean()
    stds = grouped.std()
    counts = grouped.count()
    sems = stds / np.sqrt(counts)

    fig, ax = plt.subplots(figsize=(max(6, len(means) * 2), 5))
    x = np.arange(len(means))
    ax.bar(x, means.values, yerr=1.96 * sems.values, capsize=5, alpha=0.8, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=15, ha="right")
    ax.set_title("Reward medio (con IC 95%)")
    ax.set_ylabel("Total Reward")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()

    filepath = output_dir / f"mean_reward_comparison.{format}"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Salvato: {filepath}")
    return filepath


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str | Path = "results/plots",
    format: str = "png",
) -> List[Path]:
    """
    Genera tutti i grafici disponibili.

    Args:
        df: DataFrame con i risultati.
        output_dir: Directory di output.
        format: Formato immagine.

    Returns:
        Lista di percorsi ai file generati.
    """
    if df.empty:
        logger.warning("DataFrame vuoto, nessun grafico generato.")
        return []

    paths = []
    paths.append(plot_reward_distribution(df, output_dir, format))
    paths.append(plot_reward_over_episodes(df, output_dir, format))
    paths.append(plot_mean_reward_comparison(df, output_dir, format))
    return paths
