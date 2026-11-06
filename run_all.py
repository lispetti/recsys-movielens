"""Reproduce every number and figure in the report.

Usage:
    python run_all.py --data-dir data/ml-1m
"""
import argparse
import json
from pathlib import Path

from src import explore, split
from src.data import load_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ml-1m"),
                        help="Folder containing ratings.dat, movies.dat, users.dat (default: data/ml-1m)")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="Where results/ and figures/ are written (default: current folder)")
    args = parser.parse_args()

    results_dir = args.out_dir / "results"
    figures_dir = args.out_dir / "figures"

    print(f"Loading data from {args.data_dir} ...")
    ratings, movies, users = load_all(args.data_dir)

    print("Step 1: data characterization")
    stats = explore.run(ratings, movies, users, results_dir, figures_dir)
    print(json.dumps(stats, indent=2))

    print("\nStep 2: global temporal train/validation/test split")
    data_split = split.temporal_split(ratings)
    s_stats = split.split_stats(ratings, data_split)
    with open(results_dir / "split_stats.json", "w") as f:
        json.dump(s_stats, f, indent=2)
    split.plot_split(ratings, data_split, figures_dir / "temporal_split.png")
    print(json.dumps(s_stats, indent=2))

    print(f"\nDone. Results in {results_dir}/, figures in {figures_dir}/")


if __name__ == "__main__":
    main()
