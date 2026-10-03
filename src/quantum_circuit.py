"""Quantum feature map circuit (Qiskit) and exact fidelity-kernel utilities.

Circuit (n qubits, `reps` layers):

    repeat reps times:
        RY(x_i) on every qubit i          # angle (RY) encoding of the feature vector
        CNOT(i, i+1) for i = 0..n-2       # linear entanglement

Note on `reps`: for a FIDELITY kernel |<phi(x)|phi(y)>|^2, a single RY layer followed by a
CNOT layer gives the same kernel as the un-entangled circuit (the trailing CNOT layer is a
fixed unitary and cancels). Entanglement only changes the kernel when the data are encoded
again after a CNOT layer, which is why the default is reps = 2.
"""
from __future__ import annotations

import numpy as np


class QuantumUnavailableError(RuntimeError):
    """Raised when Qiskit (or a required Qiskit component) cannot be used."""


def qiskit_status() -> tuple[bool, str]:
    try:
        import qiskit
        from qiskit.quantum_info import Statevector  # noqa: F401
        return True, getattr(qiskit, "__version__", "unknown")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def build_feature_map(n_qubits: int, reps: int = 2):
    """Return a parameterised Qiskit QuantumCircuit implementing the feature map."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
    except Exception as exc:
        raise QuantumUnavailableError(f"Qiskit is not installed or could not be imported ({exc}).") from exc
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    if reps < 1:
        raise ValueError("reps must be at least 1.")
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="QueSat_FeatureMap")
    for _ in range(reps):
        for i in range(n_qubits):
            qc.ry(x[i], i)
        qc.barrier()
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
    return qc


def ordered_parameters(circuit):
    return sorted(circuit.parameters, key=lambda p: p.index)


def bind_features(circuit, x):
    """Bind one feature vector (angles) to the circuit; returns a new circuit."""
    params = ordered_parameters(circuit)
    if len(params) != len(x):
        raise ValueError(f"Circuit expects {len(params)} features, got {len(x)}.")
    return circuit.assign_parameters({p: float(v) for p, v in zip(params, x)})


def statevectors(circuit, X) -> np.ndarray:
    """Exact statevectors |phi(x)> for every row of X (shape: n_samples x 2**n_qubits)."""
    try:
        from qiskit.quantum_info import Statevector
    except Exception as exc:
        raise QuantumUnavailableError(f"Qiskit is not installed or could not be imported ({exc}).") from exc
    X = np.asarray(X, dtype=float)
    return np.array([Statevector(bind_features(circuit, row)).data for row in X])


def fidelity_gram(SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    """Fidelity kernel matrix K[i, j] = |<phi(a_i)|phi(b_j)>|^2."""
    return np.abs(SA.conj() @ SB.T) ** 2


def circuit_text(circuit) -> str:
    return str(circuit.draw(output="text", fold=120))


def circuit_figure(circuit):
    """Matplotlib drawing of the circuit, or None if the mpl drawer is unavailable."""
    try:
        return circuit.draw(output="mpl", fold=-1, style="iqp")
    except Exception:
        return None
