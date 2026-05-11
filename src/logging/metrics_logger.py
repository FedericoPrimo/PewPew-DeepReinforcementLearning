"""
Salvataggio delle metriche degli episodi in CSV o JSON.
"""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from src.loops.run_episode import EpisodeResult

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "seed",
    "agent_name",
    "preprocessing_mode",
    "episode_index",
    "total_reward",
    "episode_length",
    "masking_enabled",
    "duration_seconds",
]


class MetricsLogger:
    """
    Logger che salva i risultati degli episodi su disco in formato CSV o JSON.

    Supporta la scrittura incrementale (riga per riga) per CSV,
    o la scrittura al flush per JSON.

    Args:
        output_dir: Directory dove salvare i file.
        run_name: Nome della run (usato nel nome del file).
        format: "csv" o "json".
    """

    def __init__(
        self,
        output_dir: str | Path = "results",
        run_name: Optional[str] = None,
        format: Literal["csv", "json"] = "csv",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"
        ext = "csv" if format == "csv" else "json"
        self.filepath = self.output_dir / f"{run_name}.{ext}"

        self._results: List[EpisodeResult] = []
        self._csv_file = None
        self._csv_writer = None

        if format == "csv":
            self._init_csv()

        logger.info(f"MetricsLogger → {self.filepath}")

    def _init_csv(self) -> None:
        self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=FIELDNAMES)
        self._csv_writer.writeheader()

    def log(self, result: EpisodeResult) -> None:
        """
        Registra il risultato di un episodio.

        Args:
            result: EpisodeResult da registrare.
        """
        self._results.append(result)

        if self.format == "csv" and self._csv_writer:
            self._csv_writer.writerow(result.to_dict())
            self._csv_file.flush()

    def flush(self) -> None:
        """Scrive tutti i risultati su disco. Necessario per JSON."""
        if self.format == "json":
            data = [r.to_dict() for r in self._results]
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        if self._csv_file:
            self._csv_file.flush()

        logger.info(f"Salvati {len(self._results)} episodi → {self.filepath}")

    def close(self) -> None:
        """Chiude i file aperti."""
        self.flush()
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def results(self) -> List[EpisodeResult]:
        return list(self._results)
