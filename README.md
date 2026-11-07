# Recommender Systems Course Project: MovieLens 1M

Team project for the Recommender Systems course (Tampere University).

> Work in progress. Part 1 (data, protocol, baselines) is being built.

## Requirements

- **Python 3.10** (developed on 3.10.6; any 3.10.x works). If you don't have it, install it from
  https://www.python.org/downloads/release/python-31011/ (installers for Windows and macOS).
  It installs alongside any other Python version you already have.
- Packages pinned in `requirements.txt`

## Setup

Run these from the repository folder. All later commands assume the virtual environment is activated.

macOS / Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

## Data

Download MovieLens 1M from GroupLens (https://grouplens.org/datasets/movielens/1m/) and extract it
into a `data` folder.

macOS / Linux:

```bash
mkdir data
curl -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
tar -xf data/ml-1m.zip -C data
```

Windows 10+ (PowerShell; note `curl.exe`, not `curl`):

```powershell
mkdir data
curl.exe -L -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
tar -xf data/ml-1m.zip -C data
```

Alternatively, download the zip in a browser and extract it into `data/`.

This creates `data/ml-1m/` containing `ratings.dat`, `movies.dat` and `users.dat`.
The dataset is not included in this repository (see GroupLens usage terms).

## Running

One command reproduces every number and figure (written to `results/` and `figures/`):

```bash
python run_all.py --data-dir data/ml-1m
```
