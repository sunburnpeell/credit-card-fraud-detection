"""Metrics, threshold selection and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def pick_threshold(y_true, scores, min_precision: float = 0.80) -> float:
    """Highest-recall threshold that still keeps precision >= min_precision.
    Fraud teams care about catching fraud without flooding analysts with false alarms."""
    p, r, t = precision_recall_curve(y_true, scores)
    p, r = p[:-1], r[:-1]
    ok = p >= min_precision
    if not ok.any():
        return float(t[np.argmax(2 * p * r / (p + r + 1e-12))])
    return float(t[ok][np.argmax(r[ok])])


def metrics_at(y_true, scores, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def plot_pr_curves(y_true, score_map: dict, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in score_map.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
    ax.axhline(y_true.mean(), ls="--", c="gray", label="Random")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(m: dict, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Fraud"],
           yticklabels=["Legit", "Fraud"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix @ {m['threshold']:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(names, values, path, top=15):
    idx = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[idx], np.array(values)[idx], color="#2b6cb0")
    ax.set(title="Top feature importances (XGBoost gain)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
