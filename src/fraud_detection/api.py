"""FastAPI service: POST /predict scores a transaction in real time.

Run: uvicorn fraud_detection.api:app --reload
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import add_features

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xgb_fraud.joblib"
app = FastAPI(title="Fraud Detection API", version="0.1.0")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run `python -m fraud_detection.train`.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Transaction(BaseModel):
    Time: float = Field(..., ge=0, description="Seconds since first transaction")
    Amount: float = Field(..., ge=0)
    features: dict[str, float] = Field(..., description="Anonymised PCA features V1..V28")


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    risk_level: str
    top_factors: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    b = get_bundle()
    missing = [f"V{i}" for i in range(1, 29) if f"V{i}" not in tx.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing[:5]}...")
    row = {"Time": tx.Time, "Amount": tx.Amount, **tx.features}
    X = add_features(pd.DataFrame([row]))[b["columns"]]
    p = float(b["model"].predict_proba(X)[0, 1])
    thr = float(b["threshold"])
    level = "high" if p >= thr else "medium" if p >= thr / 2 else "low"
    import xgboost as xgb

    contrib = b["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    top = sorted(zip(b["columns"], contrib), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    factors = [{"feature": f, "impact": round(float(c), 4)} for f, c in top]
    return Prediction(fraud_probability=p, is_fraud=p >= thr, threshold=thr,
                      risk_level=level, top_factors=factors)
