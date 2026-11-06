"""Loading MovieLens 1M from the raw ``::``-separated .dat files."""
from pathlib import Path

import pandas as pd

# movies.dat contains non-UTF-8 characters in some titles; the files are latin-1.
ENCODING = "latin-1"


def _read_dat(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",  # multi-character separator requires the python engine
        names=columns,
        encoding=ENCODING,
    )


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """user_id, item_id, rating (1-5), timestamp (unix seconds), datetime (UTC)."""
    df = _read_dat(Path(data_dir) / "ratings.dat", ["user_id", "item_id", "rating", "timestamp"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def load_movies(data_dir: Path) -> pd.DataFrame:
    """item_id, title, year, genres (list of strings)."""
    df = _read_dat(Path(data_dir) / "movies.dat", ["item_id", "title", "genres"])
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$").astype("Int64")
    df["genres"] = df["genres"].str.split("|")
    return df


def load_users(data_dir: Path) -> pd.DataFrame:
    """user_id, gender, age (band code), occupation (code), zip."""
    return _read_dat(
        Path(data_dir) / "users.dat", ["user_id", "gender", "age", "occupation", "zip"]
    )


def load_all(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    if not (data_dir / "ratings.dat").exists():
        raise FileNotFoundError(
            f"ratings.dat not found in {data_dir}. See README.md for how to download MovieLens 1M."
        )
    return load_ratings(data_dir), load_movies(data_dir), load_users(data_dir)
