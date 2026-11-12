"""Check MarkovChain against the lecture's worked example and a slow literal implementation.

Run:  python -m tests.check_markov
"""
import numpy as np
import pandas as pd

from src.markov import MarkovChain


def make_history(sessions: dict, catalogue_size: int) -> pd.DataFrame:
    rows = [(u, item, 5, t) for u, seq in sessions.items() for t, item in enumerate(seq)]
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    return df.sample(frac=1, random_state=0)  # row order must not matter, only timestamps


def lecture_example() -> None:
    """Lecture 4.1 slide 5-6: after i3 comes i1 in S3, S4 and i5 in S1, so P(i1|i3)=2/3."""
    sessions = {1: [2, 3, 5], 2: [4, 1, 5], 3: [2, 3, 1], 4: [3, 1, 5], 5: [4, 1, 3]}
    model = MarkovChain(n_last=1, beta=0).fit(make_history(sessions, 5), np.arange(1, 6))
    p = model.P_[2]  # catalogue positions: item i is column i - 1; row for i3 is 2
    assert np.isclose(p[0], 2 / 3) and np.isclose(p[4], 1 / 3) and p.sum() == 1, p
    assert np.isclose(model.P_[3][0], 1.0)  # i1 follows i4 in every session where i4 appears
    # a user whose session so far is [i2, i3] gets i1 first (not i5), then i5
    new = pd.DataFrame({"user_id": [9, 9], "item_id": [2, 3], "rating": 5, "timestamp": [0, 1]})
    model = MarkovChain(n_last=1, beta=0).fit(pd.concat([make_history(sessions, 5), new]),
                                              np.arange(1, 6))
    assert model.recommend(9, exclude={2, 3}, k=2) == [1, 5]
    # the order matters: [i3, i2] is a different observation (nothing ever follows i2 -> i3 backwards)
    rev = pd.DataFrame({"user_id": [9, 9], "item_id": [3, 2], "rating": 5, "timestamp": [0, 1]})
    model = MarkovChain(n_last=1, beta=0).fit(pd.concat([make_history(sessions, 5), rev]),
                                              np.arange(1, 6))
    assert np.isclose(model.scores(9)[2], 1.0)  # after i2 only i3 has been seen, in S1 and S3
    print("lecture example: P(i1|i3)=2/3, P(i5|i3)=1/3, order is respected")


def brute_force_scores(sessions: dict, n_items: int, n_last: int, decay: float, beta: float,
                       user: int) -> np.ndarray:
    P = np.zeros((n_items, n_items))
    for seq in sessions.values():
        for a, b in zip(seq[:-1], seq[1:]):
            P[a - 1, b - 1] += 1
    for j in range(n_items):
        total = P[j].sum() + beta
        if total > 0:
            P[j] /= total
    seq = sessions[user]
    score = np.zeros(n_items)
    for t, item in enumerate(reversed(seq[-n_last:])):
        score += decay ** t * P[item - 1]
    return score


def random_check() -> None:
    rng = np.random.default_rng(0)
    n_items = 40
    sessions = {u: list(rng.choice(np.arange(1, n_items + 1), size=rng.integers(3, 15), replace=False))
                for u in range(1, 31)}
    history = make_history(sessions, n_items)
    worst = 0.0
    for n_last in [1, 3, 8]:
        for decay in [1.0, 0.5]:
            for beta in [0, 3]:
                model = MarkovChain(n_last, decay, beta).fit(history, np.arange(1, n_items + 1))
                for u in sessions:
                    diff = np.abs(model.scores(u) - brute_force_scores(sessions, n_items, n_last,
                                                                      decay, beta, u)).max()
                    worst = max(worst, diff)
                    assert diff < 1e-12, (n_last, decay, beta, u, diff)
    print(f"MarkovChain matches the brute-force formula on all configs (max abs diff {worst:.2e})")


def tie_check() -> None:
    """Same-second ratings are ordered by their original row, whatever the frame's row order."""
    h = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [3, 1, 2], "rating": 5,
                      "timestamp": [7, 7, 7]})
    a = MarkovChain().fit(h, np.arange(1, 4))
    b = MarkovChain().fit(h.iloc[::-1], np.arange(1, 4))
    assert (a.counts_ == b.counts_).all() and a.counts_[2, 0] == 1 and a.counts_[0, 1] == 1
    print("tied timestamps: order follows the original row, independent of frame order")


if __name__ == "__main__":
    lecture_example()
    random_check()
    tie_check()
