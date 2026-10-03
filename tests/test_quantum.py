"""Quantum tests.

* Tests marked with `needs_qiskit` run only when Qiskit is installed (they test the real code).
* The NumPy reference-simulator tests always run and verify the circuit maths / kernel logic.
"""
import numpy as np
import pytest

from src import quantum_circuit as qc
from src import quantum_model as qm
from src.data_loader import FEATURES, generate_demo_data
from src.preprocessing import preprocess
from src.simulation import run_pipeline
from tests import reference_sim as ref

HAS_QISKIT = qc.qiskit_status()[0]
needs_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit not installed")


@pytest.fixture(scope="module")
def prepared():
    return preprocess(generate_demo_data("Krishna", "Water-body Change", n=120), FEATURES)


# ---------------------------------------------------------------- reference maths (always run)
def test_reference_states_are_normalised():
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = ref.statevector(rng.uniform(0, np.pi, 4), reps=2)
        assert np.isclose(np.vdot(s, s).real, 1.0)


def test_reference_kernel_properties():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, (12, 4))
    S = np.array([ref.statevector(r) for r in X])
    K = qc.fidelity_gram(S, S)
    assert np.allclose(np.diag(K), 1.0) and np.allclose(K, K.T)
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-9      # positive semi-definite


def test_single_layer_entanglement_cancels_in_fidelity_kernel():
    """Documents why reps=2 is the default: with reps=1 the CNOTs do not change the kernel."""
    rng = np.random.default_rng(2)
    X = rng.uniform(0, np.pi, (6, 4))
    S1 = np.array([ref.statevector(r, reps=1) for r in X])
    K1 = qc.fidelity_gram(S1, S1)
    product = np.prod(np.cos((X[:, None, :] - X[None, :, :]) / 2) ** 2, axis=2)
    assert np.allclose(K1, product)
    S2 = np.array([ref.statevector(r, reps=2) for r in X])
    assert not np.allclose(qc.fidelity_gram(S2, S2), product)


def test_quantum_pipeline_logic_with_reference_simulator(prepared, monkeypatch):
    """Exercise the QSVM code path with the NumPy reference in place of Qiskit statevectors."""
    monkeypatch.setattr(qc, "build_feature_map", lambda n, reps=2: object())
    monkeypatch.setattr(qc, "statevectors", ref.statevectors_for(reps=2))
    r = qm.run_quantum(prepared, reps=2)
    assert r["status"] == "ok"
    assert r["K_train"].shape == (len(prepared.y_train),) * 2
    assert r["K_test"].shape == (len(prepared.y_test), len(prepared.y_train))
    assert len(r["y_pred"]) == len(prepared.y_test)
    assert r["metrics"]["accuracy"] > 0.6


# ---------------------------------------------------------------- error handling (always run)
def test_missing_qiskit_gives_friendly_error_not_exception(prepared, monkeypatch):
    def boom(n, reps=2):
        raise qc.QuantumUnavailableError("Qiskit is not installed")
    monkeypatch.setattr(qc, "build_feature_map", boom)
    r = qm.run_quantum(prepared)
    assert r["status"] == "error" and "classical" in r["message"].lower()


def test_pipeline_keeps_classical_when_quantum_fails(monkeypatch):
    def boom(n, reps=2):
        raise qc.QuantumUnavailableError("Qiskit is not installed")
    monkeypatch.setattr(qc, "build_feature_map", boom)
    df = generate_demo_data("Krishna", "Water-body Change")
    res = run_pipeline(df, "Krishna", "Water-body Change", True)
    assert res["status"] == "partial"
    assert res["classical"]["status"] == "ok" and res["quantum"]["status"] == "error"
    assert res["scene"] is not None and list(res["scene_preds"]) == [res["classical"]["name"]]


# ---------------------------------------------------------------- real Qiskit (skipped if absent)
@needs_qiskit
def test_circuit_structure():
    c = qc.build_feature_map(4, reps=2)
    ops = c.count_ops()
    assert c.num_qubits == 4 and ops["ry"] == 8 and ops["cx"] == 6


@needs_qiskit
def test_qiskit_statevectors_match_reference():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, np.pi, (5, 4))
    c = qc.build_feature_map(4, reps=2)
    S_q = qc.statevectors(c, X)
    S_r = np.array([ref.statevector(r, 2) for r in X])
    assert np.allclose(qc.fidelity_gram(S_q, S_q), qc.fidelity_gram(S_r, S_r), atol=1e-9)


@needs_qiskit
def test_qsvm_end_to_end(prepared):
    r = qm.run_quantum(prepared, reps=2)
    assert r["status"] == "ok" and r["engine"] == "statevector"
    assert np.allclose(np.diag(r["K_train"]), 1.0)


@needs_qiskit
def test_qiskit_ml_kernel_matches_statevector():
    pytest.importorskip("qiskit_machine_learning")
    X = np.random.default_rng(4).uniform(0, np.pi, (6, 4))
    m = qm.QuantumKernelSVM(4, reps=2)
    assert np.allclose(m.kernel_matrix(X, None, "statevector"), m.kernel_matrix(X, None, "qiskit-ml"), atol=1e-6)


@needs_qiskit
def test_aer_sampling_and_transpile_summary():
    from src.simulation import aer_sample, transpile_summary
    c = qc.build_feature_map(4, reps=2)
    t = transpile_summary(c)
    assert t["status"] == "ok" and t["depth"] > 0
    pytest.importorskip("qiskit_aer")
    s = aer_sample(c, [0.3, 1.0, 2.0, 0.5], shots=256)
    assert s["status"] == "ok" and sum(s["counts"].values()) == 256
