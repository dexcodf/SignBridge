"""Load and clean the Sign Language MNIST dataset.

Dataset (Kaggle): https://www.kaggle.com/datasets/datamunge/sign-language-mnist
Each row is one example: a ``label`` column (0-24) plus ``pixel1..pixel784``
holding a 28x28 grayscale image flattened row-major, values 0-255.

Download (needs the Kaggle CLI configured with ~/.kaggle/kaggle.json):

    kaggle datasets download -d datamunge/sign-language-mnist -p data --unzip

The cleaning here is deliberately light because the dataset is already
well-formed; we mostly validate, scale and reshape so downstream code can
trust the contract defined in :mod:`config`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import config


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read one of the Sign Language MNIST CSV files into a DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the dataset first:\n"
            "  kaggle datasets download -d datamunge/sign-language-mnist "
            "-p data --unzip"
        )
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and tidy a raw Sign Language MNIST frame.

    Steps: drop exact duplicate rows, drop rows with any missing pixel,
    coerce the label to int and discard any label outside the valid range.
    Returns a fresh, index-reset DataFrame.
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna()
    df["label"] = df["label"].astype(int)
    df = df[(df["label"] >= 0) & (df["label"] < config.NUM_CLASSES)]
    return df.reset_index(drop=True)


def to_arrays(df: pd.DataFrame, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Split a cleaned frame into image tensor ``X`` and label vector ``y``.

    ``X`` has shape ``(n, IMG_SIZE, IMG_SIZE, 1)``; pixels are scaled to
    ``[0, 1]`` when ``normalize`` is True. ``y`` is an int array of labels.
    """
    y = df["label"].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    if normalize:
        # not in-place: to_numpy() may return a read-only view
        pixels = pixels / np.float32(255.0)
    X = pixels.reshape(-1, config.IMG_SIZE, config.IMG_SIZE, 1)
    return X, y


def load_dataset(
    train_csv: str | Path = config.TRAIN_CSV,
    test_csv: str | Path = config.TEST_CSV,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience loader: returns ``(X_train, y_train, X_test, y_test)``."""
    train = clean(load_raw(train_csv))
    test = clean(load_raw(test_csv))
    X_train, y_train = to_arrays(train, normalize=normalize)
    X_test, y_test = to_arrays(test, normalize=normalize)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_dataset()
    print(f"train: {Xtr.shape}  test: {Xte.shape}")
    print(f"classes present: {sorted(np.unique(ytr))}")
