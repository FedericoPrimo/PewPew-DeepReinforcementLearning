"""
Configurazione pytest: aggiunge la root del progetto al sys.path
così che gli import `from src.xxx` funzionino senza installare il package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
