from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import FEATURE_COLUMNS

_LABEL = "Label"


def load_and_clean(file_paths: list) -> tuple[pd.DataFrame, StandardScaler]:
    """Load CIC-IDS2019 CSVs, clean rows, binarize labels, and scale configured features.

    Returns the cleaned DataFrame and the StandardScaler fitted on the feature columns.
    """
    if not file_paths:
        raise ValueError("file_paths must contain at least one path")

    chunks = []
    for p in file_paths:
        for chunk in pd.read_csv(p, on_bad_lines='skip', encoding='utf-8',
                                low_memory=False, chunksize=50000):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    df.columns = df.columns.str.strip()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if _LABEL not in df.columns:
        raise ValueError(f"Missing label column {_LABEL!r}")

    labels = df[_LABEL].astype(str).str.strip()
    df[_LABEL] = np.where(labels.str.casefold() == "benign", 0, 1)

    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        warnings.warn(
            f"Skipping {len(missing)} feature(s) not in data: {missing[:5]}"
            + (" ..." if len(missing) > 5 else ""),
            stacklevel=2,
        )
    if not present:
        raise ValueError(
            "None of config.FEATURE_COLUMNS appear in the data after cleaning."
        )

    for col in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=present + [_LABEL], inplace=True)
    df.reset_index(drop=True, inplace=True)

    scaler = StandardScaler()
    df[present] = scaler.fit_transform(df[present])

    return df, scaler
