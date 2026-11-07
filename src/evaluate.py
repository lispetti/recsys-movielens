"""Top-N evaluation shared by all recommenders (Lecture 3.1, slides 7-9).

Protocol:
- Candidate set: the full catalogue (every movie in movies.dat), minus the items the user
  already rated in the history data.
- Each evaluated user gets a top-K list (K = 10).
- Relevant items: the user's ratings in the target period with rating >= threshold.
- Precision@K = hits / K
- Recall@K    = hits / number of relevant items
- NDCG@K      = DCG / ideal DCG, binary relevance, DCG = sum over hits of 1 / log2(rank + 1),
                ideal DCG = the same sum for min(#relevant, K) hits at the top
- Catalogue coverage = distinct items recommended to at least one user / catalogue size
Per-user metrics are averaged over evaluated users (every user weighs the same).
"""
from typing import Protocol

import numpy as np
import pandas as pd

from src.split import RELEVANT_THRESHOLD

K = 10


class Recommender(Protocol):
    def fit(self, history: pd.DataFrame, catalogue: np.ndarray) -> "Recommender": ...

    def recommend(self, user_id: int, exclude: set, k: int) -> list: ...


def relevant_items(target: pd.DataFrame, threshold: int = RELEVANT_THRESHOLD) -> dict:
    rel = target[target["rating"] >= threshold]
    return rel.groupby("user_id")["item_id"].agg(set).to_dict()


def seen_items(history: pd.DataFrame) -> dict:
    return history.groupby("user_id")["item_id"].agg(set).to_dict()


def user_metrics(recs: list, relevant: set, k: int = K) -> dict:
    hits = np.array([item in relevant for item in recs[:k]], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts[: len(hits)]).sum())
    idcg = float(discounts[: min(len(relevant), k)].sum())
    return {
        "precision": hits.sum() / k,
        "recall": hits.sum() / len(relevant),
        "ndcg": dcg / idcg,
    }


def evaluate(model: Recommender, history: pd.DataFrame, target: pd.DataFrame,
             eval_users: np.ndarray, catalogue: np.ndarray, k: int = K,
             threshold: int = RELEVANT_THRESHOLD) -> tuple[dict, dict]:
    """Return (averaged metrics, {user_id: top-k list})."""
    relevant = relevant_items(target, threshold)
    seen = seen_items(history)
    lists, rows = {}, []
    for u in eval_users:
        recs = model.recommend(u, seen.get(u, set()), k)
        lists[u] = recs
        rows.append(user_metrics(recs, relevant[u], k))
    per_user = pd.DataFrame(rows)
    recommended = set().union(*lists.values()) if lists else set()
    return {
        # Labels follow k, so a non-default cutoff can never be reported as a top-10 result.
        f"precision@{k}": float(per_user["precision"].mean()),
        f"recall@{k}": float(per_user["recall"].mean()),
        f"ndcg@{k}": float(per_user["ndcg"].mean()),
        "catalogue_coverage": len(recommended) / len(catalogue),
        "evaluated_users": int(len(eval_users)),
    }, lists


def check_against_sklearn(lists: dict, target: pd.DataFrame, catalogue: np.ndarray,
                          n_users: int = 200, k: int = K, seed: int = 0,
                          threshold: int = RELEVANT_THRESHOLD) -> dict:
    """Compare our NDCG@k with scikit-learn's ndcg_score on a sample of users.

    The spec asks teams that write their own metrics to show agreement with a reference.
    Our list order is turned into scores (first item highest); unrecommended items get 0.
    """
    from sklearn.metrics import ndcg_score

    relevant = relevant_items(target, threshold)
    rng = np.random.default_rng(seed)
    users = rng.choice(np.array(list(lists)), size=min(n_users, len(lists)), replace=False)
    pos = {item: i for i, item in enumerate(catalogue)}
    max_diff = 0.0
    for u in users:
        y_true = np.zeros(len(catalogue))
        y_true[[pos[i] for i in relevant[u] if i in pos]] = 1
        y_score = np.zeros(len(catalogue))
        for rank, item in enumerate(lists[u]):
            y_score[pos[item]] = k - rank
        ours = user_metrics(lists[u], relevant[u], k)["ndcg"]
        ref = ndcg_score(y_true[None, :], y_score[None, :], k=k)
        max_diff = max(max_diff, abs(ours - ref))
    return {"users_checked": int(len(users)), "k": k, "threshold": threshold,
            "max_abs_difference_ndcg": max_diff, "seed": seed}
