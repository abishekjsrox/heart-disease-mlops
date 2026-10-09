"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from heartrisk import config

log = logging.getLogger(__name__)

_CP_MAP = {"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4}
_THAL_MAP = {"normal": 3, "fixed": 6, "reversable": 7}


def _fetch(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec - fixed URLs
        return resp.read().decode("utf-8-sig")


def parse_uci(text: str) -> pd.DataFrame:
    """Parse processed.cleveland.data (no header, '?' marks missing values)."""
    return pd.read_csv(io.StringIO(text), header=None, names=config.UCI_COLUMNS,
                       na_values="?")


def parse_mirror(text: str) -> pd.DataFrame:
    """Convert the ISLR-style mirror (same 303 Cleveland rows) to the UCI layout."""
    m = pd.read_csv(io.StringIO(text))
    m = m.drop(columns=[c for c in m.columns if c.startswith("Unnamed") or c == ""])
    df = pd.DataFrame({
        "age": m["Age"], "sex": m["Sex"],
        "cp": m["ChestPain"].map(_CP_MAP),
        "trestbps": m["RestBP"], "chol": m["Chol"], "fbs": m["Fbs"],
        "restecg": m["RestECG"], "thalach": m["MaxHR"], "exang": m["ExAng"],
        "oldpeak": m["Oldpeak"], "slope": m["Slope"], "ca": m["Ca"],
        "thal": m["Thal"].map(_THAL_MAP),
        # mirror only carries presence/absence, which is all we need downstream
        "num": (m["AHD"].str.strip().str.lower() == "yes").astype(int),
    })
    return df[config.UCI_COLUMNS]


def download(dest: Path = config.RAW_FILE, force: bool = False) -> Path:
    """Download the raw dataset (UCI first, GitHub mirror as fallback)."""
    if dest.exists() and not force:
        log.info("Raw data already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = parse_uci(_fetch(config.UCI_URL))
        source = config.UCI_URL
    except Exception as exc:  # network blocked, UCI down, etc.
        log.warning("UCI download failed (%s); using mirror", exc)
        df = parse_mirror(_fetch(config.MIRROR_URL))
        source = config.MIRROR_URL
    if len(df) != 303:
        raise ValueError(f"Expected 303 Cleveland rows, got {len(df)}")
    df.to_csv(dest, index=False, na_rep="?")
    log.info("Saved %d rows from %s to %s", len(df), source, dest)
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    return pd.read_csv(path, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, validate ranges/codes, binarise the target and drop duplicates.

    Missing values are *kept* (as NaN) on purpose: they are imputed inside the
    sklearn pipeline so the imputer is fitted on training folds only (no leakage)
    and the exact same logic runs at inference time.
    """
    df = df.copy()
    for col in config.UCI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # invalid codes / out-of-range values become missing rather than silently kept
    for col, codes in config.VALID_CODES.items():
        df.loc[~df[col].isin(codes) & df[col].notna(), col] = np.nan
    for col, (lo, hi) in config.VALID_RANGES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    df[config.TARGET] = (df["num"] > 0).astype(int)  # 0 = absence, 1-4 = presence
    df = df.drop(columns=["num"]).drop_duplicates().reset_index(drop=True)
    return df


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def make_clean_dataset() -> pd.DataFrame:
    download()
    df = clean(load_raw())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_FILE, index=False)
    log.info("Clean dataset: %s rows -> %s", df.shape, config.CLEAN_FILE)
    return df
