# Recommender Systems Course Project: MovieLens 1M

Team project for the Recommender Systems course (Tampere University).

> Work in progress. Part 1 (data, protocol, baselines) is being built.

## Requirements

- Python 3.10
- Packages pinned in `requirements.txt`

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download MovieLens 1M from GroupLens: https://grouplens.org/datasets/movielens/1m/

```bash
mkdir -p data
curl -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip data/ml-1m.zip -d data
```

This creates `data/ml-1m/` containing `ratings.dat`, `movies.dat` and `users.dat`.
The dataset is not included in this repository (see GroupLens usage terms).

## Running

To be added: a single command that reproduces every number and figure in the report, e.g.

```bash
python run_all.py --data-dir data/ml-1m
```
