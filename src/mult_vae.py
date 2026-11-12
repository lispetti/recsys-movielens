"""Mult-VAE (Liang et al., WWW 2018; Lecture 3.2, slides 9-10), from the RecPack library.

Not self-implemented: the network (input -> 600 -> 200 -> 600 -> items, tanh), the input
normalisation and dropout, the multinomial likelihood + KL loss and the training loop are
RecPack's `MultVAE`. This file only
1. builds the binary user x item matrix RecPack expects from the history ratings,
2. fixes RecPack's KL-annealing schedule (see `_PaperAnnealing`),
3. exposes fit / recommend like the other models, and
4. tunes the hyperparameters on validation (fit on train, evaluate on val).

Fixed at the paper's values: hidden layer 600, latent dimension 200, input dropout 0.5,
Adam with learning rate 1e-3, batch size 500.
"""
import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from recpack.algorithms import MultVAE as RecPackMultVAE

from src import evaluate as ev
from src.split import RELEVANT_THRESHOLD, Split, evaluation_users

# RecPack logs every epoch and its validation sampling triggers a harmless scipy warning.
logging.getLogger("recpack").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

BATCH_SIZE = 500

# Search space (validation, fit on train): 2 x 3 x 2 = 12 configs, all with seed 0.
EPOCHS_GRID = [25, 50]
MAX_BETA_GRID = [0.2, 0.5, 1.0]
POSITIVE_ONLY_GRID = [False, True]  # input = every rating, or only ratings >= 4
TUNING_SEED = 0


class _PaperAnnealing(RecPackMultVAE):
    """RecPack 0.3.6 ramps beta from 0 to 1 and only then sets it to max_beta. The paper (and
    Lecture 3.2, slide 10) ramps beta from 0 upward and stops at the cap:
    beta = min(max_beta, step / anneal_steps)."""

    @property
    def _beta(self):
        return min(self.max_beta, self.steps / self.anneal_steps)


class MultVAE:
    def __init__(self, epochs: int, max_beta: float, positive_only: bool, seed: int = 0):
        self.epochs = epochs
        self.max_beta = max_beta
        self.positive_only = positive_only
        self.seed = seed

    @property
    def name(self) -> str:
        feedback = f"ratings >= {RELEVANT_THRESHOLD} only" if self.positive_only else "all ratings"
        return f"input {feedback}, {self.epochs} epochs, beta annealed to {self.max_beta}"

    def fit(self, history: pd.DataFrame, catalogue: np.ndarray) -> "MultVAE":
        self.catalogue_ = np.asarray(catalogue)
        self.users_ = np.sort(history["user_id"].unique())
        self.user_pos_ = pd.Series(np.arange(len(self.users_)), index=self.users_)
        if self.positive_only:
            history = history[history["rating"] >= RELEVANT_THRESHOLD]
        rows = self.user_pos_.loc[history["user_id"].to_numpy()].to_numpy()
        cols = np.searchsorted(self.catalogue_, history["item_id"].to_numpy())
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(self.users_), len(self.catalogue_)))

        # Anneal over the whole run, so beta reaches max_beta after a max_beta share of training.
        steps_per_epoch = int(np.ceil((X.getnnz(axis=1) > 0).sum() / BATCH_SIZE))
        model = _PaperAnnealing(batch_size=BATCH_SIZE, max_epochs=self.epochs, learning_rate=1e-3,
                                seed=self.seed, max_beta=self.max_beta,
                                anneal_steps=self.epochs * steps_per_epoch, keep_last=True,
                                validation_sample_size=1)
        # RecPack requires validation data and scores it after every epoch. We tune the number
        # of epochs ourselves (keep_last=True keeps the final model), so it only gets one user.
        model.fit(X, (X, X))
        # Users with no input interactions get an all-zero row (ties are broken by item id).
        self.scores_ = model.predict(X).toarray().astype(np.float32)
        return self

    def recommend(self, user_id: int, exclude: set, k: int) -> list:
        u = self.user_pos_.get(user_id)
        s = self.scores_[u].astype(np.float64) if u is not None else np.zeros(len(self.catalogue_))
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
    for epochs, max_beta, positive_only in itertools.product(EPOCHS_GRID, MAX_BETA_GRID,
                                                             POSITIVE_ONLY_GRID):
        model = MultVAE(epochs, max_beta, positive_only, TUNING_SEED).fit(split.train, catalogue)
        metrics, _ = ev.evaluate(model, split.train, split.val, val_users, catalogue,
                                 relevant=relevant, seen=seen)
        rows.append({"epochs": epochs, "max_beta": max_beta, "positive_only": positive_only,
                     **metrics})
        print(f"  val  {model.name:<60} NDCG@10={metrics['ndcg@10']:.4f}")
    val_df = pd.DataFrame(rows)
    val_df.to_csv(results_dir / "mult_vae_validation.csv", index=False)
    best = val_df.loc[val_df["ndcg@10"].idxmax()]
    return {"epochs": int(best["epochs"]), "max_beta": float(best["max_beta"]),
            "positive_only": bool(best["positive_only"])}