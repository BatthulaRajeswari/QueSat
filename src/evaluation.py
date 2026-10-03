"""Metrics and neutral result summaries. Every number comes from real predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

CLASSICAL_NAME = "Classical SVM (RBF)"
QUANTUM_NAME = "Quantum Kernel + QSVM"
METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_test": int(len(y_true)),
    }


def comparison_table(classical: dict | None, quantum: dict | None) -> pd.DataFrame:
    """One row per model; unavailable results are NaN (shown as 'unavailable' in the UI)."""
    rows = []
    for name, res in ((CLASSICAL_NAME, classical), (QUANTUM_NAME, quantum)):
        if res and res.get("status") == "ok":
            m = res["metrics"]
            rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                         "Recall": m["recall"], "F1": m["f1"], "Status": "ok"})
        else:
            rows.append({"Model": name, "Accuracy": np.nan, "Precision": np.nan,
                         "Recall": np.nan, "F1": np.nan, "Status": "unavailable"})
    return pd.DataFrame(rows)


def neutral_summary(classical: dict | None, quantum: dict | None) -> str:
    """Describe the measured difference without declaring a winner."""
    if not (classical and classical.get("status") == "ok"):
        return "Classical SVM result is unavailable."
    if not (quantum and quantum.get("status") == "ok"):
        return "Quantum Kernel + QSVM result is unavailable, so no comparison can be made."
    c, q = classical["metrics"], quantum["metrics"]
    diff = q["accuracy"] - c["accuracy"]
    return (f"On the same held-out test set (n = {c['n_test']}), the classical SVM reached accuracy "
            f"{c['accuracy']:.3f} and the quantum-kernel SVM reached {q['accuracy']:.3f} "
            f"(difference {diff:+.3f}; F1 {c['f1']:.3f} vs {q['f1']:.3f}). "
            "This is a single train/test split on one dataset, so it does not support a general "
            "statement about either approach.")
