"""Train baseline + XGBoost, evaluate on a held-out future slice, save artifacts.

Usage: python -m fraud_detection.train [--data data/creditcard.csv]
"""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import evaluate as ev
from .data import DEFAULT_PATH, TARGET, add_features, load_data, model_columns, time_split

ROOT = Path(__file__).resolve().parents[2]


def main(data_path=DEFAULT_PATH, min_precision=0.80):
    df = add_features(load_data(data_path))
    cols = model_columns()
    train, val, test = time_split(df)
    Xtr, ytr = train[cols], train[TARGET]
    Xva, yva = val[cols], val[TARGET]
    Xte, yte = test[cols], test[TARGET]
    print(f"train={len(train):,} val={len(val):,} test={len(test):,} | "
          f"fraud rate train={ytr.mean():.4%} test={yte.mean():.4%}")

    baseline = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000))
    baseline.fit(Xtr, ytr)

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw ** 0.5, eval_metric="aucpr",
        early_stopping_rounds=40, tree_method="hist", n_jobs=-1, random_state=42)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    scores = {
        "Logistic Regression": baseline.predict_proba(Xte)[:, 1],
        "XGBoost": xgb.predict_proba(Xte)[:, 1],
    }
    # Threshold chosen on validation only, never on test
    thr = ev.pick_threshold(yva, xgb.predict_proba(Xva)[:, 1], min_precision)
    thr_base = ev.pick_threshold(yva, baseline.predict_proba(Xva)[:, 1], min_precision)
    results = {
        "Logistic Regression": ev.metrics_at(yte, scores["Logistic Regression"], thr_base),
        "XGBoost": ev.metrics_at(yte, scores["XGBoost"], thr),
    }
    for k, v in results.items():
        print(f"{k:20s} PR-AUC={v['pr_auc']:.3f} P={v['precision']:.3f} "
              f"R={v['recall']:.3f} F1={v['f1']:.3f} (TP={v['tp']} FP={v['fp']} FN={v['fn']})")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    ev.plot_pr_curves(yte, scores, ROOT / "results" / "pr_curve.png")
    ev.plot_confusion(results["XGBoost"], ROOT / "results" / "confusion_matrix.png")
    ev.plot_importance(cols, xgb.feature_importances_, ROOT / "results" / "feature_importance.png")
    (ROOT / "results" / "metrics.json").write_text(json.dumps(results, indent=2))
    joblib.dump({"model": xgb, "threshold": thr, "columns": cols}, ROOT / "models" / "xgb_fraud.joblib")
    print("Saved model + results.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_PATH)
    ap.add_argument("--min-precision", type=float, default=0.80)
    a = ap.parse_args()
    main(a.data, a.min_precision)
