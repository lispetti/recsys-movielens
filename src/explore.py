"""Part 1: characterize the MovieLens 1M data (counts, long tail, ratings, time)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (0 = perfectly equal, 1 = maximally unequal)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def top_share(counts: np.ndarray, fraction: float) -> float:
    """Share of all interactions held by the top `fraction` of entities."""
    x = np.sort(counts)[::-1]
    k = max(1, int(round(fraction * len(x))))
    return float(x[:k].sum() / x.sum())


def describe_counts(s: pd.Series) -> dict:
    return {
        "min": int(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": int(s.max()),
    }


def compute_stats(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> dict:
    n_ratings = len(ratings)
    n_users = ratings["user_id"].nunique()
    n_items_rated = ratings["item_id"].nunique()
    n_items_catalogue = len(movies)

    per_user = ratings.groupby("user_id").size()
    per_item = ratings.groupby("item_id").size()
    item_counts = per_item.to_numpy()
    sorted_items = np.sort(item_counts)[::-1]
    items_for_half = int(np.searchsorted(np.cumsum(sorted_items), 0.5 * n_ratings) + 1)

    rating_counts = ratings["rating"].value_counts().sort_index()

    # --- time ---
    dt = ratings["datetime"]
    monthly = ratings.groupby(dt.dt.strftime("%Y-%m")).size()
    by_year = ratings.groupby(dt.dt.year).size()
    first_month_end = dt.min() + pd.DateOffset(months=1)
    six_months_end = dt.min() + pd.DateOffset(months=6)

    # How long is each user active, and how bursty?
    span_days = ratings.groupby("user_id")["timestamp"].agg(lambda t: (t.max() - t.min()) / 86400)
    first_seen = ratings.groupby("user_id")["datetime"].min()
    user_day = ratings.assign(day=dt.dt.floor("D")).groupby("user_id")["day"]
    active_days = user_day.nunique()
    first_day = user_day.transform("min")
    share_on_first_day = float((ratings.assign(day=dt.dt.floor("D"))["day"] == first_day).mean())

    return {
        "counts": {
            "ratings": n_ratings,
            "users": int(n_users),
            "users_in_users_dat": len(users),
            "items_in_catalogue": n_items_catalogue,
            "items_with_ratings": int(n_items_rated),
            "items_never_rated": int(n_items_catalogue - n_items_rated),
            "density_rated_items": n_ratings / (n_users * n_items_rated),
            "density_full_catalogue": n_ratings / (n_users * n_items_catalogue),
        },
        "ratings_per_user": describe_counts(per_user),
        "ratings_per_item": describe_counts(per_item),
        "long_tail": {
            "gini_item_popularity": gini(item_counts),
            "gini_user_activity": gini(per_user.to_numpy()),
            "share_of_ratings_top_1pct_items": top_share(item_counts, 0.01),
            "share_of_ratings_top_10pct_items": top_share(item_counts, 0.10),
            "share_of_ratings_top_20pct_items": top_share(item_counts, 0.20),
            "items_needed_for_50pct_of_ratings": items_for_half,
            "items_needed_for_50pct_of_ratings_frac": items_for_half / n_items_rated,
            "items_with_lt_5_ratings": int((per_item < 5).sum()),
            "items_with_lt_20_ratings": int((per_item < 20).sum()),
            "share_of_ratings_top_10pct_users": top_share(per_user.to_numpy(), 0.10),
        },
        "ratings": {
            "distribution": {int(k): int(v) for k, v in rating_counts.items()},
            "share": {int(k): float(v / n_ratings) for k, v in rating_counts.items()},
            "mean": float(ratings["rating"].mean()),
            "share_ge_4": float((ratings["rating"] >= 4).mean()),
        },
        "time": {
            "first": str(dt.min()),
            "last": str(dt.max()),
            "span_days": float((dt.max() - dt.min()).days),
            "ratings_per_year": {int(k): int(v) for k, v in by_year.items()},
            "busiest_month": monthly.idxmax(),
            "busiest_month_ratings": int(monthly.max()),
            "share_in_first_month": float((dt < first_month_end).mean()),
            "share_in_first_6_months": float((dt < six_months_end).mean()),
            "users_first_rating_after_2000": int((first_seen.dt.year > 2000).sum()),
            "user_active_span_days": describe_counts(span_days.round()),
            "users_active_on_single_day": int((active_days == 1).sum()),
            "users_active_on_single_day_frac": float((active_days == 1).mean()),
            "share_of_ratings_on_users_first_day": share_on_first_day,
            # Ratings whose exact second is shared with at least one other rating by the same user.
            "share_of_ratings_tied_with_another_by_same_user": float(
                ratings.duplicated(["user_id", "timestamp"], keep=False).mean()),
        },
    }


# ---------------------------------------------------------------------------- plots


def plot_long_tail(ratings: pd.DataFrame, out: Path) -> None:
    per_item = np.sort(ratings.groupby("item_id").size().to_numpy())[::-1]
    per_user = np.sort(ratings.groupby("user_id").size().to_numpy())[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(np.arange(1, len(per_item) + 1), per_item, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("Item rank (by number of ratings)")
    ax.set_ylabel("Ratings per item (log)")
    ax.set_title("Item popularity (long tail)")

    ax = axes[1]
    ax.plot(np.arange(1, len(per_user) + 1), per_user, color="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("User rank (by number of ratings)")
    ax.set_ylabel("Ratings per user (log)")
    ax.set_title("User activity")

    ax = axes[2]
    for counts, label, color in [(per_item, "items", "tab:blue"), (per_user, "users", "tab:orange")]:
        x = np.arange(1, len(counts) + 1) / len(counts)
        ax.plot(100 * x, 100 * np.cumsum(counts) / counts.sum(), label=label, color=color)
    ax.plot([0, 100], [0, 100], "k--", lw=0.8, label="perfect equality")
    ax.set_xlabel("Top x% of items / users")
    ax.set_ylabel("% of all ratings")
    ax.set_title("Concentration of ratings")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ratings(ratings: pd.DataFrame, out: Path) -> None:
    counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, 100 * counts / counts.sum(), color="tab:blue")
    ax.set_xlabel("Rating")
    ax.set_ylabel("% of ratings")
    ax.set_title("Rating distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_time(ratings: pd.DataFrame, out: Path) -> None:
    dt = ratings["datetime"].dt.tz_localize(None)
    weekly = ratings.groupby(dt.dt.to_period("W").dt.start_time).size()
    first_seen = np.sort(ratings.groupby("user_id")["datetime"].min().dt.tz_localize(None).to_numpy())
    rating_times = np.sort(dt.to_numpy())
    span_days = ratings.groupby("user_id")["timestamp"].agg(lambda t: (t.max() - t.min()) / 86400)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    ax = axes[0]
    ax.plot(weekly.index, weekly.values, color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylabel("Ratings per week (log)")
    ax.set_title("Rating activity over time")
    ax.tick_params(axis="x", labelrotation=30)

    ax = axes[1]
    ax.plot(rating_times, 100 * np.arange(1, len(rating_times) + 1) / len(rating_times),
            label="ratings entered", color="tab:blue")
    ax.plot(first_seen, 100 * np.arange(1, len(first_seen) + 1) / len(first_seen),
            label="users who have joined", color="tab:orange")
    ax.set_ylabel("% of total (cumulative)")
    ax.set_title("Cumulative ratings and users over time")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=30)

    ax = axes[2]
    x = np.sort(span_days.to_numpy())
    ax.plot(np.maximum(x, 1 / 24), 100 * np.arange(1, len(x) + 1) / len(x), color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("Days between a user's first and last rating (log; <1h shown as 1h)")
    ax.set_ylabel("% of users (cumulative)")
    ax.set_title("How long users stay active")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame,
        results_dir: Path, figures_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_stats(ratings, movies, users)
    with open(results_dir / "data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    plot_long_tail(ratings, figures_dir / "long_tail.png")
    plot_ratings(ratings, figures_dir / "rating_distribution.png")
    plot_time(ratings, figures_dir / "activity_over_time.png")
    return stats
