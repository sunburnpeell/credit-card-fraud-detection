"""Data loading and time-based splitting."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "creditcard.csv"


def load_data(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle 'Credit Card Fraud Detection' "
            "dataset (creditcard.csv) into the data/ folder."
        )
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features on top of the PCA-anonymised columns."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"].clip(lower=0))
    out["hour"] = (out["Time"] // 3600) % 24
    out["is_night"] = ((out["hour"] < 6) | (out["hour"] >= 22)).astype(int)
    return out


def model_columns() -> list[str]:
    return FEATURES + ["log_amount", "hour", "is_night"]


def time_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Chronological split. Random splits leak future patterns in fraud data,
    so we train on the past and evaluate on the future, like production."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]
