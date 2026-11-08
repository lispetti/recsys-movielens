"""Item-based kNN baseline (Lecture 1.2, slides 5-10).

Fit (offline, on the history data only):
1. Build the user x item matrix from the history ratings and transform it according to the
   similarity measure (see SIMILARITIES below).
2. Item-item similarity = cosine between the transformed item columns (Jaccard for binary
   overlap). The diagonal is zeroed so an item is never its own neighbour.
3. Optional significance weighting (slide 10): multiply sim(i, j) by min(n_ij, gamma) / gamma,
   where n_ij is the number of users who rated both i and j. gamma = 0 switches it off.
4. Neighbourhood: for every candidate item i keep only its k most similar items N_k(i).
   Negative similarities are dropped (slide 10: disagreement predicts little).

Recommend (for user u):
    score(u, i) = sum over j in N_k(i) that u rated of  sim(i, j) * w(u, j)
where w(u, j) is u's entry in the transformed matrix (1 for binary measures, the rating, or the
mean-centred rating). Items are ranked by this score over the full catalogue minus the items u
already rated; ties are broken by item id so the ranking is deterministic.

Why a sum and not the slide-5 prediction  avg(u) + sum(sim * dev) / sum(|sim|):  the
denominator turns the score into a rating estimate, which makes an item reached through one
weak neighbour score as high as an item reached through many strong ones. For top-N ranking
that pushes obscure items to the top, so ranking work (Deshpande & Karypis, 2004) uses the
unnormalised sum. The prediction is a means; the ranking is the product (slide 5).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

# name -> (how the user x item matrix is transformed, similarity on the transformed columns)
SIMILARITIES = {
    "cosine (binary)": "every rating counts as 1 (implicit); cosine between item columns",
    "jaccard": "every rating counts as 1; |users of i and j| / |users of i or j|",
    "cosine (raw ratings)": "cosine between columns of raw 1-5 ratings",
    "adjusted cosine": "ratings minus the user's mean rating (removes user rating bias); cosine",
    "pearson (item-centred)": "ratings minus the item's mean rating; cosine. Centred over all of the item's "
                              "ratings, not only the co-rated ones, so it approximates Pearson",
}


class ItemKNN:
    def __init__(self, similarity: str = "cosine (binary)", k: int = 50, shrinkage: int = 0):
        if similarity not in SIMILARITIES:
            raise ValueError(f"unknown similarity {similarity!r}; choose from {list(SIMILARITIES)}")
        self.similarity = similarity
        self.k = k
        self.shrinkage = shrinkage

    @property
    def name(self) -> str:
        shrink = f", significance weighting gamma={self.shrinkage}" if self.shrinkage else ""
        return f"{self.similarity}, k={self.k}{shrink}"

    # ---- fitting -------------------------------------------------------------------------
    def _weights(self, history: pd.DataFrame, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        r = history["rating"].to_numpy(dtype=np.float64)
        if self.similarity in ("cosine (binary)", "jaccard"):
            return np.ones_like(r)
        if self.similarity == "cosine (raw ratings)":
            return r
        if self.similarity == "adjusted cosine":
            user_mean = np.bincount(rows, weights=r) / np.bincount(rows)
            return r - user_mean[rows]
        # pearson (item-centred)
        item_mean = np.bincount(cols, weights=r, minlength=cols.max() + 1) / \
            np.maximum(np.bincount(cols, minlength=cols.max() + 1), 1)
        return r - item_mean[cols]

    def fit(self, history: pd.DataFrame, catalogue: np.ndarray,
            keep_similarities: bool = False) -> "ItemKNN":
        self.catalogue_ = np.asarray(catalogue)
        item_pos = pd.Series(np.arange(len(self.catalogue_)), index=self.catalogue_)
        self.users_ = np.sort(history["user_id"].unique())
        self.user_pos_ = pd.Series(np.arange(len(self.users_)), index=self.users_)

        rows = self.user_pos_.loc[history["user_id"].to_numpy()].to_numpy()
        cols = item_pos.loc[history["item_id"].to_numpy()].to_numpy()
        w = self._weights(history, rows, cols)
        shape = (len(self.users_), len(self.catalogue_))
        W = sp.csr_matrix((w, (rows, cols)), shape=shape)  # user profiles used for scoring
        self.W_ = W

        sim = np.ascontiguousarray((W.T @ W).toarray())  # item x item dot products (row-major)
        if self.similarity == "jaccard":
            n = np.asarray((W != 0).sum(axis=0)).ravel()
            union = n[:, None] + n[None, :] - sim
            sim = np.divide(sim, union, out=np.zeros_like(sim), where=union > 0)
        else:
            norms = np.sqrt(np.diag(sim))
            denom = np.outer(norms, norms)
            sim = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > 0)

        if self.shrinkage:
            B = sp.csr_matrix((np.ones_like(w), (rows, cols)), shape=shape)
            co_counts = (B.T @ B).toarray()
            sim *= np.minimum(co_counts, self.shrinkage) / self.shrinkage

        np.fill_diagonal(sim, 0.0)
        sim[sim < 0] = 0.0
        sim = sim.astype(np.float32)
        # Each row's items sorted by decreasing similarity; equal similarities are broken by
        # item id (stable sort), so neighbourhoods and results are deterministic.
        order = np.argsort(-sim, axis=1, kind="stable").astype(np.int32)
        self.neighbours_ = self._top_k(sim, order, self.k)
        if keep_similarities:
            self.full_similarities_, self.order_ = sim, order
        return self

    def with_k(self, k: int) -> "ItemKNN":
        """Same fitted similarities, different neighbourhood size (cheap k tuning).

        Requires fit(..., keep_similarities=True)."""
        other = ItemKNN(self.similarity, k, self.shrinkage)
        other.catalogue_, other.users_, other.user_pos_, other.W_ = \
            self.catalogue_, self.users_, self.user_pos_, self.W_
        other.neighbours_ = self._top_k(self.full_similarities_, self.order_, k)
        return other

    @staticmethod
    def _top_k(sim: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
        """Keep, for every target item i, its k largest similarities sim(i, j).

        Returned transposed and dense for fast scoring: PT[j, i] = sim(i, j) if j in N_k(i),
        else 0."""
        n = sim.shape[0]
        k = min(k, n - 1)
        idx = order[:, :k]
        PT = np.zeros(sim.shape, dtype=sim.dtype, order="C")  # row-major: scoring gathers rows
        PT[idx, np.arange(n)[:, None]] = np.take_along_axis(sim, idx, axis=1)
        return PT

    # ---- recommending --------------------------------------------------------------------
    def scores(self, user_id: int) -> np.ndarray:
        u = self.user_pos_.get(user_id)
        if u is None:
            return np.zeros(len(self.catalogue_))
        start, end = self.W_.indptr[u], self.W_.indptr[u + 1]
        rated, w = self.W_.indices[start:end], self.W_.data[start:end]
        # score(u, i) = sum_j w(u, j) * PT[j, i], over the items j that u rated
        return (w.astype(np.float32) @ self.neighbours_[rated]).astype(np.float64)

    def recommend(self, user_id: int, exclude: set, k: int) -> list:
        s = self.scores(user_id)
        if exclude:
            pos = np.searchsorted(self.catalogue_, np.fromiter(exclude, dtype=np.int64))
            s[pos] = -np.inf
        # catalogue is sorted by id, so a stable sort on -score breaks ties by item id
        order = np.argsort(-s, kind="stable")
        return [int(i) for i in self.catalogue_[order[:k]]]
