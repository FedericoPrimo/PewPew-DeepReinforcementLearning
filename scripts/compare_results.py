"""
Script per l'analisi statistica e la generazione di grafici
a partire dai risultati salvati in results/.

Uso:
    python scripts/compare_results.py
    python scripts/compare_results.py --results-dir results/
    python scripts/compare_results.py --plot-format pdf
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.compare_runs import load_results, compare_agents, compute_stats
from src.analysis.plots import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisi statistica e grafici dei risultati."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory con i file CSV/JSON dei risultati",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="results/plots",
        help="Directory di output per i grafici",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disabilita la generazione dei grafici",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
    logger = logging.getLogger("compare_results")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Directory non trovata: {results_dir}")
        sys.exit(1)

    df = load_results(results_dir)

    if df.empty:
        print("Nessun risultato trovato. Esegui prima scripts/run_dummy.py")
        sys.exit(0)

    # Statistiche
    stats = compute_stats(df)
    print("\n" + "=" * 60)
    print("STATISTICHE")
    print("=" * 60)
    print(stats.to_string(index=False))

    # Report confronto
    report = compare_agents(df)
    print("\n" + report)

    # Grafici
    if not args.no_plots:
        paths = generate_all_plots(df, output_dir=args.plots_dir, format=args.plot_format)
        print(f"\nGrafici salvati in: {args.plots_dir}")
        for p in paths:
            print(f"  → {p.name}")


if __name__ == "__main__":
    main()
