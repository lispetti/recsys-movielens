"""Writing results.csv in the course results-template format.

Column names match the "Part 1 - Baselines" tab of the results template exactly. The extra
leading "Sheet" column says which template tab a row belongs to (the template allows adding
columns), so later parts can append rows for their own tabs.
"""
from pathlib import Path

import pandas as pd

PART1_COLUMNS = ["Model", "Variant / settings", "Similarity", "k", "Precision@10", "Recall@10",
                 "NDCG@10", "Catalogue coverage", "Seed", "Notes"]


def part1_row(model: str, variant: str, metrics: dict, similarity: str = "", k="",
              seed: str = "none (deterministic)", notes: str = "") -> dict:
    return {
        "Sheet": "Part 1 - Baselines",
        "Model": model,
        "Variant / settings": variant,
        "Similarity": similarity,
        "k": k,
        "Precision@10": round(metrics["precision@10"], 6),
        "Recall@10": round(metrics["recall@10"], 6),
        "NDCG@10": round(metrics["ndcg@10"], 6),
        "Catalogue coverage": round(metrics["catalogue_coverage"], 6),
        "Seed": seed,
        "Notes": notes,
    }


def write_results(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows, columns=["Sheet"] + PART1_COLUMNS)
    df.to_csv(path, index=False)
