"""Writing results.csv in the course results-template format.

One file holds the rows of every template tab. The leading "Sheet" column says which tab a row
belongs to (the template allows adding columns); the other columns are the tabs' columns, named
exactly as in the template. A column that a row's tab does not have is left empty, so filtering
on "Sheet" gives each tab's table with the template's columns.
"""
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = {"precision@10": "Precision@10", "recall@10": "Recall@10", "ndcg@10": "NDCG@10",
                  "catalogue_coverage": "Catalogue coverage"}
_METRICS = list(METRIC_COLUMNS.values())

SHEET_COLUMNS = {
    "Part 1 - Baselines": ["Model", "Variant / settings", "Similarity", "k", *_METRICS, "Seed", "Notes"],
    "Part 2 - Models": ["Model", "Paradigm", "Self-implemented?", "Key hyperparameters", *_METRICS,
                        "Variability / CI / test", "Seed(s)", "Notes"],
    "Part 2 - Disaggregated": ["Model", "Breakdown dimension", "Group", "Definition / threshold",
                               *_METRICS, "Seed(s)", "Notes"],
    "Part 2 - Sparsity": ["Model", "Training interactions (%)", *_METRICS, "Seed(s)", "Notes"],
}


def metric_columns(metrics: dict) -> dict:
    return {column: round(metrics[key], 6) for key, column in METRIC_COLUMNS.items()}


def part1_row(model: str, variant: str, metrics: dict, similarity: str = "", k="",
              seed: str = "none (deterministic)", notes: str = "") -> dict:
    return {
        "Sheet": "Part 1 - Baselines",
        "Model": model,
        "Variant / settings": variant,
        "Similarity": similarity,
        "k": k,
        **metric_columns(metrics),
        "Seed": seed,
        "Notes": notes,
    }


def write_results(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(c for sheet in SHEET_COLUMNS.values() for c in sheet))
    pd.DataFrame(rows, columns=["Sheet"] + columns).to_csv(path, index=False)
