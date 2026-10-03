# Quantum methodology

## Feature map
For `n` qubits (default 4) and `reps` layers (default 2):

```
repeat reps times:
    RY(x_i) on qubit i, for all i        # angle encoding
    CNOT(i, i+1) for i = 0 … n-2         # linear entanglement
```

`x` is the preprocessed feature vector scaled to [0, π].

## Fidelity quantum kernel

```
k(x, y) = | ⟨φ(x) | φ(y)⟩ |²,   |φ(x)⟩ = U(x)|0…0⟩
```

The Gram matrix is passed to scikit-learn's `SVC(kernel="precomputed")`, which is the construction used by
Qiskit Machine Learning's `QSVC`.

## Engines
| Engine | How | Notes |
|---|---|---|
| `statevector` (default) | `qiskit.quantum_info.Statevector` per sample, `K = |S S†|²` | exact, fast, LOCAL SIMULATION |
| `qiskit-ml` | `qiskit_machine_learning.kernels.FidelityQuantumKernel` | slower; library default primitives |

If the selected engine fails, the app reports it and recomputes the *same kernel* with the statevector engine
(and says so). If Qiskit is unavailable, the quantum result is marked unavailable — nothing is faked.

## Why two layers by default
For a fidelity kernel, one RY layer followed by CNOTs yields exactly the same kernel as no entanglement: the
final fixed CNOT layer is a unitary applied to both states and cancels in the overlap
(`tests/test_quantum.py::test_single_layer_entanglement_cancels_in_fidelity_kernel`). Re-encoding the data
after a CNOT layer (`reps ≥ 2`) makes entanglement affect the kernel. `reps = 1` is available for
comparison.

## Decision values
The QSVM "decision value" is the SVM margin score. It is not a probability and is never shown as a
calibrated confidence.

## What this is not
* Not executed on quantum hardware.
* No quantum advantage is claimed. Simulation time on a classical computer says nothing about hardware speed.

## Verification
`tests/reference_sim.py` is an independent NumPy implementation of the same circuit. The test-suite checks
that Qiskit statevectors and kernels match it (runs when Qiskit is installed) and uses it to test the QSVM
code path in environments without Qiskit. The app itself never uses it.
