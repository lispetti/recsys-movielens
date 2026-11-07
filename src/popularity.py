"""Most Popular baseline.

Everyone gets the same ranking: items ordered by how many ratings they received in the
history data, skipping items the user has already rated. Two choices define a variant:

- window_days: None = count all history ("all-time"); N = count only ratings entered in the
  last N days before the end of the history ("recent window").
- positive_only: False = count every rating; True = count only ratings >= the relevance
  threshold (i.e. "most liked" rather than "most rated").

Ties in the count are broken by item id (ascending), so the ranking is deterministic.
Catalogue items with no counted ratings are ranked last.
"""
import numpy as np
import pandas as pd

from src.split import RELEVANT_THRESHOLD


class MostPopular:
    def __init__(self, window_days: int | None = None, positive_only: bool = False,
                 threshold: int = RELEVANT_THRESHOLD):
        self.window_days = window_days
        self.positive_only = positive_only
        self.threshold = threshold

    @property
    def name(self) -> str:
        window = "all-time" if self.window_days is None else f"last {self.window_days} days"
        feedback = f"ratings >= {self.threshold} only" if self.positive_only else "all ratings"
        return f"{window}, {feedback}"

    def fit(self, history: pd.DataFrame, catalogue: np.ndarray) -> "MostPopular":
        counted = history
        if self.window_days is not None:
            end = history["timestamp"].max()
            counted = counted[counted["timestamp"] > end - self.window_days * 86400]
        if self.positive_only:
            counted = counted[counted["rating"] >= self.threshold]
        counts = counted.groupby("item_id").size().reindex(catalogue, fill_value=0)
        order = pd.DataFrame({"item_id": counts.index, "count": counts.to_numpy()})
        order = order.sort_values(["count", "item_id"], ascending=[False, True])
        self.ranking_ = order["item_id"].to_numpy()
        self.counted_ratings_ = int(len(counted))
        return self

    def recommend(self, user_id: int, exclude: set, k: int) -> list:
        recs = []
        for item in self.ranking_:
            if item not in exclude:
                recs.append(int(item))
                if len(recs) == k:
                    break
        return recs
