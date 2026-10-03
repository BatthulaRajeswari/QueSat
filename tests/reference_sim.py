"""Independent NumPy simulator of the QueSat feature-map circuit.

Used ONLY by the tests, to (a) cross-check the Qiskit statevectors/kernel when Qiskit is
installed and (b) exercise the quantum-model logic in environments without Qiskit.
It is never used by the application as a fallback.
"""
from __future__ import annotations

import numpy as np


def _apply_1q(state, gate, qubit, n):
    # qubit 0 is the least-significant bit (Qiskit little-endian convention)
    psi = state.reshape([2] * n)
    axis = n - 1 - qubit
    psi = np.moveaxis(np.tensordot(gate, psi, axes=([1], [axis])), 0, axis)
    return psi.reshape(-1)


def _apply_cx(state, control, target, n):
    psi = state.reshape([2] * n).copy()
    ca, ta = n - 1 - control, n - 1 - target
    idx = [slice(None)] * n
    idx[ca] = 1
    sub = psi[tuple(idx)]
    t_axis = ta if ta < ca else ta - 1
    psi[tuple(idx)] = np.flip(sub, axis=t_axis)
    return psi.reshape(-1)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def statevector(x, reps=2):
    n = len(x)
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    for _ in range(reps):
        for i in range(n):
            state = _apply_1q(state, ry(x[i]), i, n)
        for i in range(n - 1):
            state = _apply_cx(state, i, i + 1, n)
    return state


def statevectors_for(reps=2):
    """Return a function with the signature of quantum_circuit.statevectors(circuit, X)."""
    def fn(circuit, X):
        return np.array([statevector(row, reps) for row in np.asarray(X, float)])
    return fn
