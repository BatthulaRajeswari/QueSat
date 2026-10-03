"""Quantum Kernel + QSVM.

The quantum kernel is the fidelity kernel of the feature map in `quantum_circuit`.
The SVM is scikit-learn's SVC with a precomputed kernel — the same construction that
Qiskit Machine Learning's QSVC uses internally.

Kernel engines:
  * "statevector": exact Qiskit statevector simulation (fast; default).
  * "qiskit-ml"  : qiskit_machine_learning FidelityQuantumKernel (slower; library default sampler).
If the selected engine fails, the failure is reported — the result is never faked.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.svm import SVC

from . import quantum_circuit as qc
from .evaluation import QUANTUM_NAME, compute_metrics
from .preprocessing import PreparedData
from .quantum_circuit import QuantumUnavailableError

ENGINES = {"statevector": "Qiskit exact statevector (fast)",
           "qiskit-ml": "Qiskit Machine Learning FidelityQuantumKernel (slower)"}


class QuantumKernelSVM:
    def __init__(self, n_qubits: int, reps: int = 2, C: float = 1.0, engine: str = "statevector"):
        if engine not in ENGINES:
            raise ValueError(f"Unknown kernel engine '{engine}'.")
        self.n_qubits, self.reps, self.C, self.engine = n_qubits, reps, C, engine
        self.circuit = qc.build_feature_map(n_qubits, reps)
        self.svc: SVC | None = None
        self.X_train_ = None
        self.K_train_ = None

    # ---- kernels
    def _kernel_statevector(self, XA, XB):
        SA = qc.statevectors(self.circuit, XA)
        SB = SA if XB is None else qc.statevectors(self.circuit, XB)
        return qc.fidelity_gram(SA, SB)

    def _kernel_qiskit_ml(self, XA, XB):
        try:
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
        except Exception as exc:
            raise QuantumUnavailableError(f"qiskit-machine-learning is not available ({exc}).") from exc
        kernel = FidelityQuantumKernel(feature_map=self.circuit)
        return np.asarray(kernel.evaluate(x_vec=np.asarray(XA), y_vec=None if XB is None else np.asarray(XB)))

    def kernel_matrix(self, XA, XB=None, engine: str | None = None) -> np.ndarray:
        engine = engine or self.engine
        if engine == "qiskit-ml":
            return self._kernel_qiskit_ml(XA, XB)
        return self._kernel_statevector(XA, XB)

    # ---- model
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.X_train_ = X
        self.K_train_ = self.kernel_matrix(X, None)
        self.svc = SVC(kernel="precomputed", C=self.C).fit(self.K_train_, np.asarray(y).astype(int))
        return self

    def _check(self):
        if self.svc is None:
            raise RuntimeError("Model is not fitted.")

    def predict(self, X, engine: str | None = None) -> np.ndarray:
        self._check()
        return self.svc.predict(self.kernel_matrix(X, self.X_train_, engine))

    def decision_function(self, X, engine: str | None = None) -> np.ndarray:
        self._check()
        return self.svc.decision_function(self.kernel_matrix(X, self.X_train_, engine))


def run_quantum(prepared: PreparedData, reps: int = 2, C: float = 1.0,
                engine: str = "statevector") -> dict:
    """Train/evaluate the QSVM. Returns a result dict; on failure status == 'error'."""
    t0 = time.perf_counter()
    note = None
    try:
        try:
            model = QuantumKernelSVM(prepared.n_components, reps, C, engine).fit(
                prepared.X_train, prepared.y_train)
        except Exception as exc:
            if engine == "statevector":
                raise
            note = (f"Engine '{engine}' failed ({type(exc).__name__}: {exc}). "
                    "The same fidelity kernel was recomputed with the exact Qiskit statevector engine.")
            engine = "statevector"
            model = QuantumKernelSVM(prepared.n_components, reps, C, engine).fit(
                prepared.X_train, prepared.y_train)

        K_test = model.kernel_matrix(prepared.X_test, model.X_train_)
        y_pred = model.svc.predict(K_test)
        decision = model.svc.decision_function(K_test)
        return {
            "status": "ok", "name": QUANTUM_NAME, "model": model, "engine": engine,
            "engine_note": note, "reps": reps,
            "y_pred": y_pred, "decision": decision,
            "K_train": model.K_train_, "K_test": K_test,
            "metrics": compute_metrics(prepared.y_test, y_pred),
            "fit_seconds": time.perf_counter() - t0,
        }
    except QuantumUnavailableError as exc:
        return _error(f"Quantum Kernel execution could not be completed. The classical baseline "
                      f"remains available. Check the Qiskit installation. ({exc})")
    except Exception as exc:
        return _error(f"Quantum Kernel execution could not be completed. The classical baseline "
                      f"remains available. ({type(exc).__name__}: {exc})")


def _error(message: str) -> dict:
    return {"status": "error", "name": QUANTUM_NAME, "metrics": None, "message": message}
