"""
Confronto statistico tra run/modelli diversi.

Legge file CSV o JSON dalla directory results e calcola:
- reward medio, mediano, deviazione standard
- intervalli di confidenza
- confronto tra agenti/modalità
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """
    Carica tutti i file CSV e JSON dalla directory results.

    Args:
        results_dir: Percorso alla directory con i risultati.

    Returns:
        DataFrame con tutti i risultati combinati.
    """
    results_dir = Path(results_dir)
    dfs = []

    for csv_file in results_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name
        dfs.append(df)
        logger.debug(f"Caricato {csv_file.name}: {len(df)} righe")

    for json_file in results_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["source_file"] = json_file.name
        dfs.append(df)
        logger.debug(f"Caricato {json_file.name}: {len(df)} righe")

    if not dfs:
        logger.warning(f"Nessun risultato trovato in {results_dir}")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Caricati {len(combined)} episodi da {len(dfs)} file")
    return combined


def compute_stats(
    df: pd.DataFrame,
    group_by: List[str] = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Calcola statistiche descrittive per gruppo.

    Args:
        df: DataFrame con colonna `total_reward`.
        group_by: Colonne per il raggruppamento (es. ["agent_name", "preprocessing_mode"]).
        confidence_level: Livello di confidenza per l'intervallo (default 0.95).

    Returns:
        DataFrame con statistiche per gruppo.
    """
    if df.empty:
        return pd.DataFrame()

    group_by = group_by or ["agent_name", "preprocessing_mode"]

    def agg_stats(series: pd.Series) -> pd.Series:
        n = len(series)
        mean = series.mean()
        std = series.std()
        median = series.median()
        se = std / np.sqrt(n) if n > 1 else 0.0
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df=max(n - 1, 1))
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se

        return pd.Series({
            "n_episodes": n,
            "mean_reward": round(mean, 2),
            "median_reward": round(median, 2),
            "std_reward": round(std, 2),
            "min_reward": round(series.min(), 2),
            "max_reward": round(series.max(), 2),
            f"ci_{int(confidence_level*100)}_low": round(ci_low, 2),
            f"ci_{int(confidence_level*100)}_high": round(ci_high, 2),
        })

    result = df.groupby(group_by)["total_reward"].apply(agg_stats).unstack()
    return result.reset_index()


def compare_agents(
    df: pd.DataFrame,
    agent_col: str = "agent_name",
    mode_col: str = "preprocessing_mode",
) -> str:
    """
    Genera un report testuale di confronto tra agenti/modalità.

    Args:
        df: DataFrame con i risultati.
        agent_col: Colonna con il nome dell'agente.
        mode_col: Colonna con la modalità di preprocessing.

    Returns:
        Stringa con il report formattato.
    """
    if df.empty:
        return "Nessun dato disponibile per il confronto."

    stats_df = compute_stats(df, group_by=[agent_col, mode_col])

    lines = ["=" * 60, "CONFRONTO AGENTI / MODALITÀ", "=" * 60]
    for _, row in stats_df.iterrows():
        lines.append(
            f"\n{row[agent_col]} | {row[mode_col]}\n"
            f"  Episodi: {int(row['n_episodes'])}\n"
            f"  Reward medio:   {row['mean_reward']:.2f} ± {row['std_reward']:.2f}\n"
            f"  Reward mediano: {row['median_reward']:.2f}\n"
            f"  Range:          [{row['min_reward']:.1f}, {row['max_reward']:.1f}]\n"
            f"  CI 95%:         [{row['ci_95_low']:.2f}, {row['ci_95_high']:.2f}]"
        )
    lines.append("=" * 60)
    return "\n".join(lines)
