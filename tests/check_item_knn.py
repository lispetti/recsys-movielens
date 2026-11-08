"""Check ItemKNN against a slow, literal implementation of the formula on random data.

Run:  python -m tests.check_item_knn
"""
import numpy as np
import pandas as pd

from src.item_knn import SIMILARITIES, ItemKNN


def brute_force_scores(R: np.ndarray, similarity: str, k: int, shrinkage: int) -> np.ndarray:
    """R: dense users x items rating matrix (0 = unrated). Returns users x items scores."""
    rated = R > 0
    if similarity in ("cosine (binary)", "jaccard"):
        W = rated.astype(float)
    elif similarity == "cosine (raw ratings)":
        W = R.astype(float)
    elif similarity == "adjusted cosine":
        mu = np.array([R[u, rated[u]].mean() for u in range(len(R))])
        W = np.where(rated, R - mu[:, None], 0.0)
    else:  # pearson (item-centred)
        mu = np.array([R[rated[:, i], i].mean() if rated[:, i].any() else 0 for i in range(R.shape[1])])
        W = np.where(rated, R - mu[None, :], 0.0)
    n_items = R.shape[1]
    S = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            if similarity == "jaccard":
                union = (rated[:, i] | rated[:, j]).sum()
                s = (rated[:, i] & rated[:, j]).sum() / union if union else 0.0
            else:
                d = np.linalg.norm(W[:, i]) * np.linalg.norm(W[:, j])
                s = W[:, i] @ W[:, j] / d if d else 0.0
            if shrinkage:
                s *= min((rated[:, i] & rated[:, j]).sum(), shrinkage) / shrinkage
            S[i, j] = max(s, 0.0)
    scores = np.zeros_like(W)
    for i in range(n_items):
        neigh = np.argsort(-S[i], kind="stable")[:k]
        for u in range(len(R)):
            scores[u, i] = sum(S[i, j] * W[u, j] for j in neigh)
    return scores


def main() -> None:
    rng = np.random.default_rng(0)
    n_users, n_items = 30, 25
    R = rng.integers(1, 6, size=(n_users, n_items)) * (rng.random((n_users, n_items)) < 0.3)
    u, i = np.nonzero(R)
    history = pd.DataFrame({"user_id": u + 1, "item_id": i + 1, "rating": R[u, i], "timestamp": 0})
    catalogue = np.arange(1, n_items + 1)
    worst = 0.0
    for similarity in SIMILARITIES:
        for k in [3, 10]:
            for shrinkage in [0, 5]:
                model = ItemKNN(similarity, k, shrinkage).fit(history, catalogue)
                ours = np.vstack([model.scores(uid) for uid in range(1, n_users + 1)])
                ref = brute_force_scores(R, similarity, k, shrinkage)
                diff = np.abs(ours - ref).max()
                worst = max(worst, diff)
                assert diff < 1e-4, (similarity, k, shrinkage, diff)
    print(f"ItemKNN matches the brute-force formula on all configs (max abs diff {worst:.2e})")


if __name__ == "__main__":
    main()
