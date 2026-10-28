"""Classification metrics used in Tables 2–3."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Pick the probability threshold that maximises F1 (tuned on validation data)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def classification_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_hat = (np.asarray(p) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_hat) * 100),
        "precision": float(precision_score(y_true, y_hat, zero_division=0) * 100),
        "recall": float(recall_score(y_true, y_hat, zero_division=0) * 100),
        "f1": float(f1_score(y_true, y_hat, zero_division=0) * 100),
        "auc_roc": float(roc_auc_score(y_true, p) * 100),
        "pr_auc": float(average_precision_score(y_true, p) * 100),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def summarize_folds(fold_metrics: list[dict], keys=("accuracy", "precision", "recall", "f1", "auc_roc")) -> dict:
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out
