import numpy as np
import pandas as pd

from fraud_detection.data import add_features, model_columns, time_split
from fraud_detection.evaluate import metrics_at, pick_threshold


def _df(n=1000):
    rng = np.random.default_rng(0)
    d = {"Time": np.sort(rng.uniform(0, 172800, n)), "Amount": rng.exponential(50, n)}
    for i in range(1, 29):
        d[f"V{i}"] = rng.normal(size=n)
    d["Class"] = (rng.random(n) < 0.02).astype(int)
    return pd.DataFrame(d)


def test_features_added():
    out = add_features(_df())
    assert set(model_columns()) <= set(out.columns)
    assert out["hour"].between(0, 23).all()


def test_time_split_is_chronological():
    tr, va, te = time_split(_df())
    assert tr["Time"].max() <= va["Time"].min() <= te["Time"].min()
    assert len(tr) + len(va) + len(te) == 1000


def test_threshold_respects_precision():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.05).astype(int)
    s = y * rng.uniform(0.4, 1, 5000) + (1 - y) * rng.uniform(0, 0.6, 5000)
    t = pick_threshold(y, s, 0.8)
    assert metrics_at(y, s, t)["precision"] >= 0.8
