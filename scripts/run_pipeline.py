"""Run the full dengue forecasting experiment.

Usage (from the project root, virtual environment activated):
    python scripts/run_pipeline.py            # everything, incl. figures
    python scripts/run_pipeline.py --no-figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from dengue.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figures", action="store_true", help="skip writing PNG figures")
    args = parser.parse_args()
    summary = run(make_figures=not args.no_figures)
    print("\nDone. Selected model:", summary["selected_model"])
    print("See reports/results/RESULTS.md and reports/figures/")


if __name__ == "__main__":
    main()
