"""Matrix factorization fitted with ALS (Lecture 2.1, slides 4, 7-9), from the `implicit` library.

Not self-implemented: the factorization and the alternating least squares updates are
`implicit`'s `AlternatingLeastSquares` (Hu, Koren & Volinsky, ICDM 2008: implicit-feedback ALS).
This file only
1. builds the user x item matrix `implicit` expects from the history ratings,
2. exposes fit / recommend like the other models, and
3. tunes the hyperparameters on validation (fit on train, evaluate on val).

The model in one paragraph: every user u gets a vector p_u and every item i a vector q_i, both
of length `factors`. The predicted preference is the dot product p_u . q_i. Training minimises
    sum over all (u, i) of c_ui * (x_ui - p_u . q_i)^2  +  regularization * (|p_u|^2 + |q_i|^2)
where x_ui = 1 if u rated i (else 0) and the confidence c_ui = 1 + alpha * x_ui, so rated items
count more than unrated ones. ALS alternates: fix all q and solve for every p_u exactly (a
regularised least-squares problem), then fix all p and solve for every q_i, and repeat.
Unlike the slide-4 model there are no bias terms: with 0/1 input, item popularity is learned
by the factors themselves.

Fixed: 15 ALS iterations (the library default), one thread (so a seed gives the same result
on every run).
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

from src import evaluate as ev
from src.split import RELEVANT_THRESHOLD, Split, evaluation_users

ITERATIONS = 15

# Search space (validation, fit on train): 3 x 3 x 3 x 2 = 54 configs, all with seed 0.
FACTORS_GRID = [32, 64, 128]
REGULARIZATION_GRID = [0.01, 0.1, 1.0]
ALPHA_GRID = [1.0, 10.0, 40.0]
POSITIVE_ONLY_GRID = [False, True]  # input = every rating, or only ratings >= 4
TUNING_SEED = 0


class ALS:
    def __init__(self, factors: int, regularization: float, alpha: float, positive_only: bool,
                 seed: int = 0):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.positive_only = positive_only
        self.seed = seed

    @property
    def name(self) -> str:
        feedback = f"ratings >= {RELEVANT_THRESHOLD} only" if self.positive_only else "all ratings"
        return (f"input {feedback}, {self.factors} factors, "
                f"regularization {self.regularization:g}, alpha {self.alpha:g}")

    def fit(self, history: pd.DataFrame, catalogue: np.ndarray) -> "ALS":
        self.catalogue_ = np.asarray(catalogue)
        self.users_ = np.sort(history["user_id"].unique())
        self.user_pos_ = pd.Series(np.arange(len(self.users_)), index=self.users_)
        if self.positive_only:
            history = history[history["rating"] >= RELEVANT_THRESHOLD]
        # 0/1 matrix: row = user, column = item (position in the sorted catalogue)
        rows = self.user_pos_.loc[history["user_id"].to_numpy()].to_numpy()
        cols = np.searchsorted(self.catalogue_, history["item_id"].to_numpy())
        X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(self.users_), len(self.catalogue_)))

        # implicit asks for single-threaded BLAS (it parallelises by itself; nested threads are slow)
        with threadpool_limits(1, "blas"):
            model = AlternatingLeastSquares(factors=self.factors,
                                            regularization=self.regularization, alpha=self.alpha,
                                            iterations=ITERATIONS, random_state=self.seed,
                                            num_threads=1)
            model.fit(X, show_progress=False)
        self.user_factors_ = model.user_factors  # one row p_u per user
        self.item_factors_ = model.item_factors  # one row q_i per catalogue item
        return self

    def recommend(self, user_id: int, exclude: set, k: int) -> list:
        u = self.user_pos_.get(user_id)
        if u is None:
            s = np.zeros(len(self.catalogue_))
        else:
            s = (self.item_factors_ @ self.user_factors_[u]).astype(np.float64)  # p_u . q_i
        if exclude:
            s[np.searchsorted(self.catalogue_, np.fromiter(exclude, dtype=np.int64))] = -np.inf
        order = np.argsort(-s, kind="stable")  # catalogue sorted by id: ties broken by item id
        return [int(i) for i in self.catalogue_[order[:k]]]


def tune(movies: pd.DataFrame, split: Split, results_dir: Path) -> dict:
    """Validation grid; returns the configuration with the highest validation NDCG@10."""
    catalogue = np.sort(movies["item_id"].to_numpy())
    val_users, _ = evaluation_users(split.train, split.val)
    relevant, seen = ev.relevant_items(split.val), ev.seen_items(split.train)
    rows = []
    for factors, regularization, alpha, positive_only in itertools.product(
            FACTORS_GRID, REGULARIZATION_GRID, ALPHA_GRID, POSITIVE_ONLY_GRID):
        model = ALS(factors, regularization, alpha, positive_only, TUNING_SEED)
        model.fit(split.train, catalogue)
        metrics, _ = ev.evaluate(model, split.train, split.val, val_users, catalogue,
                                 relevant=relevant, seen=seen)
        rows.append({"factors": factors, "regularization": regularization, "alpha": alpha,
                     "positive_only": positive_only, **metrics})
        print(f"  val  {model.name:<70} NDCG@10={metrics['ndcg@10']:.4f}")
    val_df = pd.DataFrame(rows)
    val_df.to_csv(results_dir / "als_validation.csv", index=False)
    best = val_df.loc[val_df["ndcg@10"].idxmax()]
    return {"factors": int(best["factors"]), "regularization": float(best["regularization"]),
            "alpha": float(best["alpha"]), "positive_only": bool(best["positive_only"])}
