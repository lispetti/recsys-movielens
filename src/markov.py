"""Markov-chain sequential recommender (Lecture 4.1, slides 6-7). SELF-IMPLEMENTED.

Written from the lecture slides with plain numpy; no recommender library is used.

Fit (one counting pass over the history, no training):
1. Put every user's history ratings in time order. This sequence is the "session" of the
   slides; a user's whole history is one sequence.
2. Count transitions: C[j, i] = number of times item i directly followed item j in a sequence.
3. P(i | j) = C[j, i] / (sum_i' C[j, i'] + beta). With beta = 0 this is exactly the slide-6
   estimate (P(i1 | i3) = 2/3 in the slide's example). beta > 0 shrinks the estimate for items j
   that were seen few times, so one lucky transition does not give P = 1.

Recommend (for user u with sequence h_1 ... h_m):
    score(u, i) = sum_{t=1..n} decay^(t-1) * P(i | h_{m-t+1})
n = 1 is the plain first-order chain of slide 6: only the last item counts. n > 1 adds the
previous items, weighted down by `decay`; it is the cheapest answer to slide 7 (one step of
memory is not enough) that stays a counting model. n and decay are tuned on validation.
Items the user already rated are excluded by the evaluation protocol (slide 7 notes that
repeat consumption is sometimes legitimate; for movie ratings it is not). Ties in score are
broken by popularity in the history and then by item id, so a user whose last items have no
observed transitions gets the Most Popular ranking, and the ranking is deterministic.

Order of ratings made in the same second: ML-1M records timestamps in whole seconds and 53% of
consecutive ratings of a user share one, so their true order is unknown. They are ordered by
their row in ratings.dat (the DataFrame index), which is not time-sorted and so carries no
order information; this is a fixed arbitrary order, not a signal. The same rule is applied to
every user, to training and to the sequence the model recommends from.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.split import Split, evaluation_users

# Search space (validation, fit on train): n_last = 1 has no decay to tune, so
# 5 betas + 5 n_last x 3 decays x 5 betas = 80 configs. Deterministic: no seed.
N_LAST_GRID = [1, 3, 5, 10, 20, 50]
DECAY_GRID = [1.0, 0.7, 0.5]  # weight of the t-th most recent item = decay^(t-1)
BETA_GRID = [0, 5, 20, 100, 500]


class MarkovChain:
    def __init__(self, n_last: int = 1, decay: float = 1.0, beta: float = 0.0):
        if n_last < 1:
            raise ValueError("n_last must be >= 1")
        self.n_last = n_last
        self.decay = decay
        self.beta = beta

    @property
    def name(self) -> str:
        if self.n_last == 1:
            memory = "first-order (last item only)"
        else:
            memory = f"last {self.n_last} items, decay {self.decay:g}"
        return f"{memory}, shrinkage beta={self.beta:g}"

    def fit(self, history: pd.DataFrame, catalogue: np.ndarray) -> "MarkovChain":
        self.catalogue_ = np.asarray(catalogue)
        n_items = len(self.catalogue_)
        users = history["user_id"].to_numpy()
        items = np.searchsorted(self.catalogue_, history["item_id"].to_numpy())
        # time order per user; ties by original row of ratings.dat (see module docstring)
        order = np.lexsort((history.index.to_numpy(), history["timestamp"].to_numpy(), users))
        users, items = users[order], items[order]

        # a transition links two neighbouring ratings of the same user
        same_user = users[1:] == users[:-1]
        prev, nxt = items[:-1][same_user], items[1:][same_user]
        counts = np.bincount(prev * n_items + nxt, minlength=n_items * n_items)
        counts = counts.reshape(n_items, n_items).astype(np.float64)
        self.counts_ = counts
        self.P_ = self._probabilities(counts, self.beta)

        # each user's sequence, as positions in the catalogue
        starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
        self.sequences_ = {int(u): s for u, s in zip(users[starts], np.split(items, starts[1:]))}
        # tie-break rank: most popular first, then lowest item id
        popularity = np.bincount(items, minlength=n_items)
        self.popularity_order_ = np.argsort(-popularity, kind="stable")
        self.popularity_rank_ = np.empty(n_items, dtype=np.int64)
        self.popularity_rank_[self.popularity_order_] = np.arange(n_items)
        return self

    @staticmethod
    def _probabilities(counts: np.ndarray, beta: float) -> np.ndarray:
        """Row j holds P(. | j) = C[j, .] / (sum of row j + beta)."""
        total = counts.sum(axis=1, keepdims=True) + beta
        return np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)

    def with_params(self, n_last: int, decay: float, beta: float) -> "MarkovChain":
        """Same fitted transition counts and sequences, other hyperparameters (cheap tuning)."""
        other = MarkovChain(n_last, decay, beta)
        fitted = {k: v for k, v in self.__dict__.items() if k.endswith("_")}
        other.__dict__.update(fitted)
        other.P_ = self._probabilities(self.counts_, beta)
        return other

    def scores(self, user_id: int) -> np.ndarray:
        seq = self.sequences_.get(user_id)
        if seq is None:
            return np.zeros(len(self.catalogue_))
        last = seq[::-1][: self.n_last]  # most recent item first
        weights = self.decay ** np.arange(len(last))
        return weights @ self.P_[last]

    def recommend(self, user_id: int, exclude: set, k: int) -> list:
        s = self.scores(user_id)
        if exclude:
            s[np.searchsorted(self.catalogue_, np.fromiter(exclude, dtype=np.int64))] = -np.inf
        # highest score first; equal scores by popularity rank (which is unique, so no further tie)
        order = np.lexsort((self.popularity_rank_, -s))
        return [int(i) for i in self.catalogue_[order[:k]]]


def tune(movies: pd.DataFrame, split: Split, results_dir: Path, figures_dir: Path) -> dict:
    """Validation grid; returns the configuration with the highest validation NDCG@10."""
    catalogue = np.sort(movies["item_id"].to_numpy())
    val_users, _ = evaluation_users(split.train, split.val)
    relevant, seen = ev.relevant_items(split.val), ev.seen_items(split.train)
    base = MarkovChain().fit(split.train, catalogue)  # counted once; beta is applied per config
    rows = []
    for n_last, decay, beta in itertools.product(N_LAST_GRID, DECAY_GRID, BETA_GRID):
        if n_last == 1 and decay != DECAY_GRID[0]:
            continue  # decay does nothing with a single item
        model = base.with_params(n_last, decay, beta)
        metrics, _ = ev.evaluate(model, split.train, split.val, val_users, catalogue,
                                 relevant=relevant, seen=seen)
        rows.append({"n_last": n_last, "decay": decay, "beta": beta, **metrics})
        print(f"  val  {model.name:<60} NDCG@10={metrics['ndcg@10']:.4f}")
    val_df = pd.DataFrame(rows)
    val_df.to_csv(results_dir / "markov_validation.csv", index=False)
    plot_validation(val_df, results_dir, figures_dir / "markov_validation.png")
    best = val_df.loc[val_df["ndcg@10"].idxmax()]
    return {"n_last": int(best["n_last"]), "decay": float(best["decay"]), "beta": float(best["beta"])}


def plot_validation(val_df: pd.DataFrame, results_dir: Path, out: Path) -> None:
    """Validation NDCG@10 against memory length n (decay 1), one line per beta, with the best
    validation NDCG@10 of the two Part 1 baselines (their CSVs are written before this runs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    full_memory = val_df[val_df["decay"] == DECAY_GRID[0]]
    for beta, g in full_memory.groupby("beta"):
        g = g.sort_values("n_last")
        ax.plot(g["n_last"], g["ndcg@10"], marker="o", ms=3, label=f"beta={beta:g}")
    for name, file in [("Most Popular", "most_popular_validation.csv"),
                       ("Item-based kNN", "item_knn_validation.csv")]:
        if (results_dir / file).exists():
            best = pd.read_csv(results_dir / file)["ndcg@10"].max()
            ax.axhline(best, color="grey", ls="--" if name == "Most Popular" else ":", lw=1,
                       label=f"{name} (best on validation)")
    ax.set_xscale("log")
    ax.set_xticks(N_LAST_GRID)
    ax.set_xticklabels([str(n) for n in N_LAST_GRID])
    ax.set_xlabel("Memory n (last items used, decay 1; n=1 is the first-order chain)")
    ax.set_ylabel("NDCG@10 on validation")
    ax.set_title("Markov chain tuning (fit on train, evaluated on validation)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
