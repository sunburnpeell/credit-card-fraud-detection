# Fraud Detection AI

Real-time credit card fraud detection: an XGBoost model on the highly imbalanced
[Kaggle / ULB credit card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 0.17% fraud), served through a FastAPI endpoint that also explains each decision.

## Why it's built this way
- **Time-based split (70/15/15)**, not random. Fraud patterns drift, so I train on the past and test on the future, like production.
- **PR-AUC, not accuracy.** A model that says "never fraud" is 99.8% accurate. PR-AUC, precision and recall are what matter.
- **Threshold tuned on validation only**: highest recall while keeping precision >= 80%, so analysts aren't flooded with false alarms. Test data is never used for tuning.
- **Baseline vs. model**: Logistic Regression compared against XGBoost.
- **Explainable**: each prediction returns its top 3 contributing features (XGBoost per-prediction contributions).
- Runs fine on a laptop CPU (trains in about a minute).

## Results (held-out future slice, 42,722 transactions, 52 frauds)

| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.721 | 0.844 | 0.731 | 0.784 |
| **XGBoost** | **0.757** | 0.765 | 0.750 | 0.757 |

XGBoost ranks fraud better (higher PR-AUC). Caveat: the test slice has only 52 fraud cases, so the
precision/recall figures are noisy; one or two transactions move them several points.

![PR curve](results/pr_curve.png)
![Confusion matrix](results/confusion_matrix.png)

## Quickstart
```bash
pip install -r requirements.txt
# put creditcard.csv in data/ (download from Kaggle)
make train      # trains, evaluates, saves model + plots to results/
make test
make serve      # http://127.0.0.1:8000/docs
```

Example request:
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"Time": 1000, "Amount": 25.5, "features": {"V1": 0, "V2": 0, ... "V28": 0}}'
```

## Structure
```
src/fraud_detection/  data.py  evaluate.py  train.py  api.py
tests/                unit + API tests
results/              metrics.json and plots
```

## Next steps
- Cost-based threshold (missed fraud costs more than a false alarm)
- LightGBM / probability calibration, SHAP dashboards
- Drift monitoring and a Docker image
- Try the mobile-money PaySim dataset

*Dataset features V1-V28 are PCA-anonymised, so feature-level interpretation is limited.*
