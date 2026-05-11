"""
Caricamento e gestione della configurazione YAML.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Wrapper attorno a un dizionario di configurazione.
    Permette accesso con notazione puntata: config.env.id
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: str | Path) -> Config:
    """
    Carica un file YAML e restituisce un oggetto Config.

    Args:
        path: Percorso al file YAML.

    Returns:
        Oggetto Config con i parametri.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(data)


def merge_configs(base: Config, overrides: Dict[str, Any]) -> Config:
    """
    Sovrascrive valori della configurazione base con un dizionario di override.
    Solo i valori di primo livello vengono sovrascritti (shallow merge).

    Args:
        base: Configurazione base.
        overrides: Dizionario con i valori da sovrascrivere.

    Returns:
        Nuova Config con i valori aggiornati.
    """
    base_dict = base.to_dict()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key].update(value)
        else:
            base_dict[key] = value
    return Config(base_dict)
