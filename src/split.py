"""Part 1: global temporal train/validation/test split (Lecture 3.1, slide 4).

One pair of cut timestamps is shared by all users: everything before the first cut is
training data, everything between the cuts is validation, everything after is test.
Cut points are placed at the 80% and 90% quantiles of the rating timestamps, so the
three parts hold roughly 80% / 10% / 10% of all ratings.

Ratings that share the exact timestamp of a cut go to the later part, so ties are never
split across two parts. The split is deterministic (no randomness, so no seed needed).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# A test item counts as relevant if the user rated it at least this high (feedback treatment).
RELEVANT_THRESHOLD = 4


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cut_train_val: int  # unix seconds; train = timestamp < cut_train_val
    cut_val_test: int   # val = cut_train_val <= timestamp < cut_val_test; test = the rest
    train_frac: float   # the fractions the cuts were computed from, so exports report the real config
    val_frac: float


def temporal_split(ratings: pd.DataFrame, train_frac: float = TRAIN_FRAC,
                   val_frac: float = VAL_FRAC) -> Split:
    ts = np.sort(ratings["timestamp"].to_numpy())
    n = len(ts)
    cut1 = int(ts[int(train_frac * n)])
    cut2 = int(ts[int((train_frac + val_frac) * n)])
    t = ratings["timestamp"]
    return Split(
        train=ratings[t < cut1],
        val=ratings[(t >= cut1) & (t < cut2)],
        test=ratings[t >= cut2],
        cut_train_val=cut1,
        cut_val_test=cut2,
        train_frac=train_frac,
        val_frac=val_frac,
    )


def evaluation_users(history: pd.DataFrame, target: pd.DataFrame,
                     threshold: int = RELEVANT_THRESHOLD) -> tuple[np.ndarray, dict]:
    """Users who can be scored on `target` given the data in `history`.

    A user is scored only if they (a) have at least one rating in `history` (otherwise
    collaborative filtering has nothing to recommend from: cold start) and (b) have at
    least one relevant rating in `target` (otherwise recall is undefined).
    Returns the user ids and counts of who was excluded and why.
    """
    target_users = target["user_id"].unique()
    known = np.isin(target_users, history["user_id"].unique())
    has_relevant = np.isin(target_users, target.loc[target["rating"] >= threshold, "user_id"].unique())
    keep = known & has_relevant
    return target_users[keep], {
        "users_in_period": int(len(target_users)),
        "excluded_cold_start": int((~known).sum()),
        "excluded_no_relevant_item": int((known & ~has_relevant).sum()),
        "evaluated_users": int(keep.sum()),
    }


def _part_summary(df: pd.DataFrame) -> dict:
    return {
        "ratings": int(len(df)),
        "share_of_all": None,  # filled in by split_stats
        "users": int(df["user_id"].nunique()),
        "items": int(df["item_id"].nunique()),
        "first": str(pd.to_datetime(df["timestamp"].min(), unit="s", utc=True)),
        "last": str(pd.to_datetime(df["timestamp"].max(), unit="s", utc=True)),
    }


def shift_stats(history: pd.DataFrame, target: pd.DataFrame, eval_users: np.ndarray,
                threshold: int = RELEVANT_THRESHOLD) -> dict:
    """How the evaluated users and their target ratings differ from the training data."""
    hist_per_user = history.groupby("user_id").size()
    hist_pop = history.groupby("item_id").size()
    top10_items = hist_pop.sort_values(ascending=False).index[: max(1, int(0.1 * len(hist_pop)))]
    tgt = target[target["user_id"].isin(eval_users)]

    def pop_of(df: pd.DataFrame) -> pd.Series:
        return df["item_id"].map(hist_pop).fillna(0)

    return {
        "history_ratings_per_user_all_users_median": float(hist_per_user.median()),
        "history_ratings_per_user_evaluated_users_median": float(hist_per_user.loc[eval_users].median()),
        "mean_rating_history": float(history["rating"].mean()),
        "mean_rating_target": float(tgt["rating"].mean()),
        "share_relevant_history": float((history["rating"] >= threshold).mean()),
        "share_relevant_target": float((tgt["rating"] >= threshold).mean()),
        "median_item_popularity_history_ratings": float(pop_of(history).median()),
        "median_item_popularity_target_ratings": float(pop_of(tgt).median()),
        "share_on_top10pct_items_history": float(history["item_id"].isin(top10_items).mean()),
        "share_on_top10pct_items_target": float(tgt["item_id"].isin(top10_items).mean()),
        "target_ratings_on_items_unseen_in_history": int((~tgt["item_id"].isin(hist_pop.index)).sum()),
        "target_relevant_on_items_unseen_in_history": int(
            ((~tgt["item_id"].isin(hist_pop.index)) & (tgt["rating"] >= threshold)).sum()),
        "target_ratings_per_evaluated_user_median": float(tgt.groupby("user_id").size().median()),
    }


def plot_split(ratings: pd.DataFrame, split: Split, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = ratings["datetime"].dt.tz_localize(None)
    weekly = ratings.groupby(dt.dt.to_period("W").dt.start_time).size()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    c1 = pd.to_datetime(split.cut_train_val, unit="s")
    c2 = pd.to_datetime(split.cut_val_test, unit="s")
    start, end = dt.min(), dt.max()
    ax.axvspan(start, c1, color="tab:green", alpha=0.15, label=f"train (until {c1:%Y-%m-%d})")
    ax.axvspan(c1, c2, color="tab:orange", alpha=0.35, label=f"validation ({c1:%Y-%m-%d} to {c2:%Y-%m-%d})")
    ax.axvspan(c2, end, color="tab:red", alpha=0.12, label=f"test (from {c2:%Y-%m-%d})")
    ax.plot(weekly.index, weekly.values, color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylabel("Ratings per week (log)")
    ax.legend(loc="upper right", fontsize=8)
    test_frac = 1 - split.train_frac - split.val_frac
    ax.set_title(f"Global temporal split ({split.train_frac:.0%} / {split.val_frac:.0%} / {test_frac:.0%} of ratings)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def split_stats(ratings: pd.DataFrame, split: Split, threshold: int = RELEVANT_THRESHOLD) -> dict:
    n = len(ratings)
    parts = {name: _part_summary(df) for name, df in
             [("train", split.train), ("val", split.val), ("test", split.test)]}
    for name, df in [("train", split.train), ("val", split.val), ("test", split.test)]:
        parts[name]["share_of_all"] = len(df) / n

    train_val = pd.concat([split.train, split.val])
    val_users, val_counts = evaluation_users(split.train, split.val, threshold)
    test_users, test_counts = evaluation_users(train_val, split.test, threshold)
    return {
        "protocol": {
            "type": "global temporal split",
            "train_frac": split.train_frac,
            "val_frac": split.val_frac,
            "test_frac": round(1 - split.train_frac - split.val_frac, 10),
            "cut_train_val": str(pd.to_datetime(split.cut_train_val, unit="s", utc=True)),
            "cut_val_test": str(pd.to_datetime(split.cut_val_test, unit="s", utc=True)),
            "relevant_threshold": threshold,
            "tuning": "fit on train, evaluate on val",
            "final": "refit on train+val, evaluate on test",
        },
        "parts": parts,
        "validation_users": val_counts,
        "test_users": test_counts,
        "shift_train_vs_val": shift_stats(split.train, split.val, val_users, threshold),
        "shift_trainval_vs_test": shift_stats(train_val, split.test, test_users, threshold),
    }
