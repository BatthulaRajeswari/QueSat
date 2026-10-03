"""Simulation helpers and the end-to-end analysis pipeline."""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from . import quantum_circuit as qc
from . import regional_visualization as rv
from .classical_model import run_classical
from .data_loader import TARGET, has_geo
from .evaluation import CLASSICAL_NAME, QUANTUM_NAME
from .feature_engineering import add_spectral_indices, feature_columns
from .preprocessing import preprocess
from .quantum_model import run_quantum

LOCAL_SIM_LABEL = "LOCAL SIMULATION"


@lru_cache(maxsize=1)
def environment_status() -> dict:
    """Report what is actually installed. Nothing here claims a remote connection."""
    ok, version = qc.qiskit_status()
    status = {"qiskit": {"available": ok, "version": version if ok else None,
                         "error": None if ok else version}}
    try:
        import qiskit_aer
        status["aer"] = {"available": True, "version": getattr(qiskit_aer, "__version__", "unknown")}
    except Exception as exc:
        status["aer"] = {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import qiskit_machine_learning
        status["qml"] = {"available": True, "version": getattr(qiskit_machine_learning, "__version__", "unknown")}
    except Exception as exc:
        status["qml"] = {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    status["simulator_ready"] = ok
    return status


def circuit_info(circuit, reps: int | None = None) -> dict:
    ops = {k: int(v) for k, v in circuit.count_ops().items() if k != "barrier"}
    return {
        "qubits": circuit.num_qubits, "depth": circuit.depth(), "gate_count": sum(ops.values()),
        "gate_breakdown": ops, "parameters": circuit.num_parameters,
        "reps": reps, "mode": LOCAL_SIM_LABEL,
    }


def aer_sample(circuit, x, shots: int = 1024, seed: int = 7) -> dict:
    """Shot-based sampling of the bound feature-map circuit on the local Aer simulator."""
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:
        return {"status": "error", "message": f"Qiskit Aer is not available ({type(exc).__name__}: {exc})."}
    try:
        bound = qc.bind_features(circuit, x)
        bound.measure_all()
        backend = AerSimulator(seed_simulator=seed)
        counts = backend.run(transpile(bound, backend), shots=shots).result().get_counts()
        return {"status": "ok", "counts": dict(counts), "shots": shots, "mode": LOCAL_SIM_LABEL}
    except Exception as exc:
        return {"status": "error", "message": f"Aer sampling failed ({type(exc).__name__}: {exc})."}


def transpile_summary(circuit) -> dict:
    """Transpile to a generic IBM-style basis (rz, sx, x, cx). No backend is contacted."""
    try:
        from qiskit import transpile
        t = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
        ops = {k: int(v) for k, v in t.count_ops().items() if k != "barrier"}
        return {"status": "ok", "depth": t.depth(), "gate_count": sum(ops.values()),
                "gate_breakdown": ops, "basis": ["rz", "sx", "x", "cx"]}
    except Exception as exc:
        return {"status": "error", "message": f"Transpilation preview failed ({type(exc).__name__}: {exc})."}


def run_pipeline(df, region: str, scenario: str, is_synthetic: bool, n_components: int | None = None,
                 test_size: float = 0.25, seed: int = 42, reps: int = 2, engine: str = "statevector",
                 use_indices: bool = False, max_samples: int = 500, build_scene: bool = True) -> dict:
    """Run preprocessing, classical SVM, quantum kernel SVM and regional classification.

    Each model fails independently: an error in the quantum path never removes the classical result.
    """
    result = {"region": region, "scenario": scenario, "is_synthetic": is_synthetic,
              "warnings": [], "timings": {}, "status": "failed", "message": None,
              "classical": None, "quantum": None, "circuit": None,
              "scene": None, "scene_preds": {}, "scene_errors": {}, "geo": None,
              "config": {"n_components": n_components, "test_size": test_size, "seed": seed,
                         "reps": reps, "engine": engine, "use_indices": use_indices}}
    t_all = time.perf_counter()
    try:
        work = df.reset_index(drop=True)
        result["n_samples_input"] = int(len(work))
        if len(work) > max_samples:
            from sklearn.model_selection import train_test_split
            keep, _ = train_test_split(np.arange(len(work)), train_size=max_samples,
                                       random_state=seed, stratify=work[TARGET])
            work = work.iloc[np.sort(keep)].reset_index(drop=True)
            result["warnings"].append(f"Dataset subsampled (stratified) to {max_samples} rows to keep "
                                      "quantum-kernel simulation fast.")
        if use_indices:
            work = add_spectral_indices(work)
        cols = feature_columns(use_indices)
        result["feature_cols"] = cols
        result["n_samples"] = int(len(work))
        result["data"] = work

        t0 = time.perf_counter()
        prepared = preprocess(work, cols, n_components, test_size, seed)
        result["prepared"] = prepared
        result["timings"]["preprocessing"] = time.perf_counter() - t0
        result["warnings"].extend(n for n in prepared.notes if n.startswith("Requested"))
    except Exception as exc:
        result["message"] = f"Preprocessing failed ({type(exc).__name__}: {exc})."
        return result

    result["classical"] = run_classical(prepared)

    try:
        circuit = qc.build_feature_map(prepared.n_components, reps)
        result["circuit"] = {"status": "ok", "circuit": circuit, "info": circuit_info(circuit, reps)}
    except Exception as exc:
        result["circuit"] = {"status": "error", "message": f"Quantum circuit could not be built ({exc})."}

    if result["circuit"]["status"] == "ok":
        result["quantum"] = run_quantum(prepared, reps=reps, engine=engine)
    else:
        result["quantum"] = {"status": "error", "name": QUANTUM_NAME, "metrics": None,
                             "message": "Quantum Kernel execution could not be completed. The classical "
                                        "baseline remains available. Check the Qiskit installation. "
                                        f"({result['circuit']['message']})"}
    result["timings"]["total"] = time.perf_counter() - t_all

    predictors = {}
    if result["classical"]["status"] == "ok":
        predictors[CLASSICAL_NAME] = result["classical"]["model"].predict
    if result["quantum"]["status"] == "ok":
        qmodel = result["quantum"]["model"]
        predictors[QUANTUM_NAME] = lambda X, m=qmodel: m.predict(X, engine="statevector")

    if predictors and is_synthetic and build_scene:
        try:
            scene = rv.build_scene(region, scenario)
            preds, errs = rv.classify_scene(scene, prepared, cols, use_indices, predictors)
            result["scene"], result["scene_preds"], result["scene_errors"] = scene, preds, errs
        except Exception as exc:
            result["warnings"].append(f"Illustrative regional scene could not be generated ({exc}).")
    elif predictors and not is_synthetic and has_geo(work):
        try:
            X_all = prepared.transform(work[cols].to_numpy(dtype=float))
            geo_preds = {}
            for name, fn in predictors.items():
                geo_preds[name] = np.asarray(fn(X_all)).astype(int)
            result["geo"] = {"df": work[["Latitude", "Longitude", TARGET]].copy(), "preds": geo_preds}
        except Exception as exc:
            result["warnings"].append(f"Coordinate map could not be generated ({exc}).")

    ok_c = result["classical"]["status"] == "ok"
    ok_q = result["quantum"]["status"] == "ok"
    result["status"] = "complete" if (ok_c and ok_q) else ("partial" if (ok_c or ok_q) else "failed")
    return result
