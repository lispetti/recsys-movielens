"""Part 1: run the Most Popular baseline variants (tune on validation, report on test)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.popularity import MostPopular
from src.results import part1_row
from src.split import Split, evaluation_users

# Recent-window lengths tried on validation. The best one (by NDCG@10) is used on test.
WINDOW_DAYS_GRID = [1, 3, 7, 14, 30, 90, 180]


def run_most_popular(movies: pd.DataFrame, split: Split, results_dir: Path) -> list[dict]:
    catalogue = np.sort(movies["item_id"].to_numpy())
    train_val = pd.concat([split.train, split.val])
    val_users, _ = evaluation_users(split.train, split.val)
    test_users, _ = evaluation_users(train_val, split.test)

    # --- validation: all-time variants + every window length, for both feedback treatments
    val_rows = []
    for positive_only in [False, True]:
        for window in [None] + WINDOW_DAYS_GRID:
            model = MostPopular(window_days=window, positive_only=positive_only).fit(split.train, catalogue)
            metrics, _ = ev.evaluate(model, split.train, split.val, val_users, catalogue)
            val_rows.append({"variant": model.name, "window_days": window, "positive_only": positive_only,
                             "counted_ratings": model.counted_ratings_, **metrics})
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(results_dir / "most_popular_validation.csv", index=False)

    best_window = {}
    for positive_only in [False, True]:
        windowed = val_df[(val_df["positive_only"] == positive_only) & val_df["window_days"].notna()]
        best_window[positive_only] = int(windowed.loc[windowed["ndcg@10"].idxmax(), "window_days"])

    # --- test: refit on train+val, evaluate once
    rows, test_details, check = [], [], None
    for positive_only in [False, True]:
        for window in [None, best_window[positive_only]]:
            model = MostPopular(window_days=window, positive_only=positive_only).fit(train_val, catalogue)
            metrics, lists = ev.evaluate(model, train_val, split.test, test_users, catalogue)
            notes = ("window length chosen on validation from "
                     f"{WINDOW_DAYS_GRID} days by NDCG@10" if window is not None else "")
            rows.append(part1_row("Most Popular", model.name, metrics, similarity="n/a", k="n/a", notes=notes))
            test_details.append({"variant": model.name, "counted_ratings": model.counted_ratings_,
                                 "top10_unpersonalised": [int(i) for i in model.ranking_[:10]], **metrics})
            if check is None:
                check = ev.check_against_sklearn(lists, split.test, catalogue)

    with open(results_dir / "most_popular_test.json", "w") as f:
        json.dump({"best_window_days": {"all_ratings": best_window[False],
                                        "positive_only": best_window[True]},
                   "candidate_set": f"full catalogue ({len(catalogue)} movies) minus items the user already rated",
                   "variants": test_details}, f, indent=2)
    with open(results_dir / "metric_check.json", "w") as f:
        json.dump({"ndcg_vs_sklearn": check}, f, indent=2)
    return rows
