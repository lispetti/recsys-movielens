"""Part 1: run the baselines (tune on validation, report on test).

- Most Popular: window length tuned on validation, four variants reported.
- Item-based kNN: similarity measure, k and significance weighting tuned on validation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.item_knn import SIMILARITIES, ItemKNN
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


# ---------------------------------------------------------------------------------------------
# Item-based kNN (Lecture 1.2, slides 5-10)
# ---------------------------------------------------------------------------------------------

# Search space, all evaluated on validation (fit on train). Full grid = 5 x 9 x 2 = 90 configs.
KNN_SIMILARITIES = list(SIMILARITIES)
KNN_K_GRID = [10, 20, 50, 100, 200, 400, 800, 1600, 3200]  # 3200 ~ no pruning (3662 items in train)
KNN_SHRINKAGE_GRID = [0, 25]  # 0 = off; 25 = significance weighting min(n_ij, 25) / 25 (slide 10)


def run_item_knn(movies: pd.DataFrame, split: Split, results_dir: Path, figures_dir: Path) -> list[dict]:
    val_df = tune_item_knn(movies, split, KNN_SIMILARITIES)
    val_df.to_csv(results_dir / "item_knn_validation.csv", index=False)
    plot_knn_validation(val_df, figures_dir / "item_knn_validation.png")
    return test_item_knn(movies, split, val_df, results_dir)


def tune_item_knn(movies: pd.DataFrame, split: Split, similarities: list[str]) -> pd.DataFrame:
    """Validation grid (fit on train, evaluate on validation) for the given similarity measures."""
    catalogue = np.sort(movies["item_id"].to_numpy())
    val_users, _ = evaluation_users(split.train, split.val)
    val_relevant, val_seen = ev.relevant_items(split.val), ev.seen_items(split.train)

    # Similarities are computed once per (measure, shrinkage); k is then varied cheaply.
    val_rows = []
    for similarity in similarities:
        for shrinkage in KNN_SHRINKAGE_GRID:
            base = ItemKNN(similarity, KNN_K_GRID[0], shrinkage).fit(split.train, catalogue,
                                                                     keep_similarities=True)
            for k in KNN_K_GRID:
                model = base if k == base.k else base.with_k(k)
                metrics, _ = ev.evaluate(model, split.train, split.val, val_users, catalogue,
                                         relevant=val_relevant, seen=val_seen)
                val_rows.append({"similarity": similarity, "k": k, "shrinkage": shrinkage, **metrics})
                print(f"  val  {model.name:<60} NDCG@10={metrics['ndcg@10']:.4f}")
            del base
    return pd.DataFrame(val_rows)


def test_item_knn(movies: pd.DataFrame, split: Split, val_df: pd.DataFrame, results_dir: Path) -> list[dict]:
    """Refit each similarity's best validation configuration on train+val; evaluate once on test."""
    catalogue = np.sort(movies["item_id"].to_numpy())
    train_val = pd.concat([split.train, split.val])
    test_users, _ = evaluation_users(train_val, split.test)
    test_relevant, test_seen = ev.relevant_items(split.test), ev.seen_items(train_val)

    # Best (k, shrinkage) per similarity measure, and the overall best, all by validation NDCG@10.
    best_per_sim = val_df.loc[val_df.groupby("similarity", sort=False)["ndcg@10"].idxmax()]
    overall = val_df.loc[val_df["ndcg@10"].idxmax()]

    rows, details = [], []
    for _, cfg in best_per_sim.iterrows():
        model = ItemKNN(cfg["similarity"], int(cfg["k"]), int(cfg["shrinkage"])).fit(train_val, catalogue)
        metrics, _ = ev.evaluate(model, train_val, split.test, test_users, catalogue,
                                 relevant=test_relevant, seen=test_seen)
        is_best = cfg["similarity"] == overall["similarity"]
        shrink = (f"significance weighting gamma={int(cfg['shrinkage'])}" if cfg["shrinkage"]
                  else "no significance weighting")
        notes = (f"k and significance weighting chosen on validation by NDCG@10 (k grid {KNN_K_GRID}, "
                 f"gamma grid {KNN_SHRINKAGE_GRID}); validation NDCG@10={cfg['ndcg@10']:.4f}")
        if is_best:
            notes = "BEST item-kNN configuration on validation (selected model). " + notes
        rows.append(part1_row("Item-based kNN", f"{shrink}; score = sum of neighbour similarities x "
                              "user's (transformed) rating; negative similarities dropped",
                              metrics, similarity=cfg["similarity"], k=int(cfg["k"]), notes=notes))
        details.append({"similarity": cfg["similarity"], "k": int(cfg["k"]), "shrinkage": int(cfg["shrinkage"]),
                        "selected_on_validation": bool(is_best), "validation_ndcg@10": float(cfg["ndcg@10"]),
                        **{f"test_{m}": v for m, v in metrics.items()}})
        print(f"  test {model.name:<60} NDCG@10={metrics['ndcg@10']:.4f}")

    with open(results_dir / "item_knn_test.json", "w") as f:
        json.dump({"search_space": {"similarity": KNN_SIMILARITIES, "k": KNN_K_GRID,
                                    "shrinkage_gamma": KNN_SHRINKAGE_GRID,
                                    "configs_evaluated_on_validation": int(len(val_df)),
                                    "selection": "highest NDCG@10 on validation (fit on train)"},
                   "similarity_definitions": SIMILARITIES,
                   "best_per_similarity": details}, f, indent=2)
    return rows


def plot_knn_validation(val_df: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (similarity, g) in zip(colors, val_df.groupby("similarity", sort=False)):
        for shrinkage, style in [(0, "-"), (KNN_SHRINKAGE_GRID[-1], "--")]:
            h = g[g["shrinkage"] == shrinkage].sort_values("k")
            label = similarity if shrinkage == 0 else None
            ax.plot(h["k"], h["ndcg@10"], style, marker="o", ms=3, color=color, label=label)
    ax.plot([], [], "k-", label="no significance weighting")
    ax.plot([], [], "k--", label=f"significance weighting gamma={KNN_SHRINKAGE_GRID[-1]}")
    ax.set_xscale("log")
    ax.set_xticks(KNN_K_GRID)
    ax.set_xticklabels([str(k) for k in KNN_K_GRID])
    ax.set_xlabel("Neighbourhood size k (log scale)")
    ax.set_ylabel("NDCG@10 on validation")
    ax.set_title("Item-based kNN tuning (fit on train, evaluated on validation)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
