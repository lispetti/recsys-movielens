"""Part 2: compare all models on test (Lecture 3.1, slides 9-10).

Every model keeps the hyperparameters it got on validation (fit on train), is refit on
train+val and evaluated once on test with the Part 1 protocol. This gives three tables:

- Overall. Stochastic models are trained with each of SEEDS; we report the mean and a 95%
  t-interval over seeds. Deterministic models are run once.
- Disaggregated, along two dimensions:
  * user activity: test users with <= / > the median number of train+val ratings (median taken
    over the evaluated test users);
  * test period: test ratings before / from the median test timestamp, each half evaluated as a
    target of its own.
- Sparsity ablation: train+val ratings subsampled uniformly at random to 50% and 20%
  (SUBSAMPLE_SEED). Hyperparameters, test users and the excluded already-rated items stay as at
  100%, so only the training data changes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src import evaluate as ev
from src.item_knn import ItemKNN
from src.mult_vae import MultVAE
from src.popularity import MostPopular
from src.results import METRIC_COLUMNS, metric_columns
from src.split import Split, evaluation_users

SEEDS = [0, 1, 2]
TRAIN_SHARES = [100, 50, 20]  # % of train+val ratings used for training
SUBSAMPLE_SEED = 0
OVERALL = ("overall", "all test users", "")  # key for the full test set (not a disaggregated row)


@dataclass
class Model:
    label: str
    paradigm: str
    make: Callable  # seed (None for deterministic models) -> unfitted model
    seeds: list
    self_implemented: bool = False


def selected_models(results_dir: Path, vae_config: dict) -> list[Model]:
    """The validation-selected variant of each Part 1 baseline, and Mult-VAE."""
    pop = pd.read_csv(results_dir / "most_popular_validation.csv")
    pop = pop.loc[pop["ndcg@10"].idxmax()]
    window = None if pd.isna(pop["window_days"]) else int(pop["window_days"])
    knn = pd.read_csv(results_dir / "item_knn_validation.csv")
    knn = knn.loc[knn["ndcg@10"].idxmax()]
    return [
        Model("Most Popular", "Non-personalised (baseline)",
              lambda seed: MostPopular(window, bool(pop["positive_only"])), [None]),
        Model("Item-based kNN", "Neighbourhood CF (baseline)",
              lambda seed: ItemKNN(knn["similarity"], int(knn["k"]), int(knn["shrinkage"])), [None]),
        Model("Mult-VAE (RecPack)", "Neural",
              lambda seed: MultVAE(**vae_config, seed=seed), SEEDS),
    ]


def test_groups(history: pd.DataFrame, test: pd.DataFrame, users: np.ndarray) -> dict:
    """(dimension, group, definition) -> (target ratings, users evaluated on them)."""
    activity = history.groupby("user_id").size().loc[users]
    cut = activity.median()
    t_cut = test["timestamp"].median()
    day = f"{pd.to_datetime(t_cut, unit='s', utc=True):%Y-%m-%d %H:%M} UTC"
    early, late = test[test["timestamp"] < t_cut], test[test["timestamp"] >= t_cut]
    return {
        OVERALL: (test, users),
        ("User activity", "low", f"<= {cut:g} train+val ratings (median of test users)"):
            (test, activity.index[activity <= cut].to_numpy()),
        ("User activity", "high", f"> {cut:g} train+val ratings"):
            (test, activity.index[activity > cut].to_numpy()),
        ("Test period", "early", f"test ratings before {day} (median test timestamp)"):
            (early, evaluation_users(history, early)[0]),
        ("Test period", "late", f"test ratings from {day} on"):
            (late, evaluation_users(history, late)[0]),
    }


def run(models: list[Model], movies: pd.DataFrame, split: Split) -> list[dict]:
    catalogue = np.sort(movies["item_id"].to_numpy())
    history = pd.concat([split.train, split.val])
    users, _ = evaluation_users(history, split.test)
    groups = test_groups(history, split.test, users)
    rows = []
    for m in models:
        for share in TRAIN_SHARES:
            train = history if share == 100 else history.sample(frac=share / 100,
                                                                 random_state=SUBSAMPLE_SEED)
            evaluated = groups if share == 100 else {OVERALL: groups[OVERALL]}
            runs = []  # per seed: group -> metrics
            for seed in m.seeds:
                model = m.make(seed).fit(train, catalogue)
                runs.append({g: ev.evaluate(model, history, target, g_users, catalogue)[0]
                             for g, (target, g_users) in evaluated.items()})
            print(f"  test {m.label:<20} {share:>3}% NDCG@10="
                  f"{np.mean([r[OVERALL]['ndcg@10'] for r in runs]):.4f}")
            rows += _rows(m, share, runs)
    return rows


def _rows(m: Model, share: int, runs: list[dict]) -> list[dict]:
    seeds = "none (deterministic)" if m.seeds == [None] else ", ".join(map(str, m.seeds))
    rows = []
    for group in runs[0]:
        per_seed = {key: [r[group][key] for r in runs] for key in METRIC_COLUMNS}
        metrics = metric_columns({key: np.mean(v) for key, v in per_seed.items()})
        dimension, name, definition = group
        if share < 100:
            rows.append({"Sheet": "Part 2 - Sparsity", "Model": m.label,
                         "Training interactions (%)": share, **metrics, "Seed(s)": seeds,
                         "Notes": f"train+val ratings subsampled uniformly at random "
                                  f"(random_state={SUBSAMPLE_SEED}); hyperparameters, test users "
                                  "and excluded items as at 100%"})
        elif group == OVERALL:
            rows.append({"Sheet": "Part 2 - Sparsity", "Model": m.label,
                         "Training interactions (%)": share, **metrics, "Seed(s)": seeds,
                         "Notes": "full train+val"})
            rows.append({"Sheet": "Part 2 - Models", "Model": m.label, "Paradigm": m.paradigm,
                         "Self-implemented?": "Yes" if m.self_implemented else "No",
                         "Key hyperparameters": m.make(m.seeds[0]).name, **metrics,
                         "Variability / CI / test": _variability(per_seed), "Seed(s)": seeds,
                         "Notes": "hyperparameters selected on validation by NDCG@10, "
                                  "refit on train+val"})
        else:
            rows.append({"Sheet": "Part 2 - Disaggregated", "Model": m.label,
                         "Breakdown dimension": dimension, "Group": name,
                         "Definition / threshold": definition, **metrics, "Seed(s)": seeds,
                         "Notes": "" if len(runs) == 1 else f"mean over {len(runs)} seeds"})
    return rows


def _variability(per_seed: dict) -> str:
    n = len(per_seed["ndcg@10"])
    if n == 1:
        return "deterministic: one run, no seed variance"
    t = stats.t.ppf(0.975, n - 1)
    intervals = []
    for key in ["precision@10", "recall@10", "ndcg@10"]:
        v = np.array(per_seed[key])
        half = t * v.std(ddof=1) / np.sqrt(n)
        intervals.append(f"{METRIC_COLUMNS[key]} [{v.mean() - half:.4f}, {v.mean() + half:.4f}]")
    return f"mean of {n} seeds; 95% t-interval: " + ", ".join(intervals)