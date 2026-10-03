"""Classical baseline: scikit-learn SVM on the shared processed feature representation."""
from __future__ import annotations

import time

from sklearn.svm import SVC

from .evaluation import CLASSICAL_NAME, compute_metrics
from .preprocessing import PreparedData


def run_classical(prepared: PreparedData, C: float = 1.0) -> dict:
    """Train and evaluate the RBF SVM. Returns a result dict (status 'ok' or 'error')."""
    try:
        t0 = time.perf_counter()
        model = SVC(kernel="rbf", C=C, gamma="scale")
        model.fit(prepared.X_train, prepared.y_train)
        y_pred = model.predict(prepared.X_test)
        decision = model.decision_function(prepared.X_test)
        return {
            "status": "ok", "name": CLASSICAL_NAME, "model": model,
            "y_pred": y_pred, "decision": decision,
            "metrics": compute_metrics(prepared.y_test, y_pred),
            "fit_seconds": time.perf_counter() - t0,
        }
    except Exception as exc:
        return {"status": "error", "name": CLASSICAL_NAME, "metrics": None,
                "message": f"Classical SVM could not be completed ({type(exc).__name__}: {exc})."}
