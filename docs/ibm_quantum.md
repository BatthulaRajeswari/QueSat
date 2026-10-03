# IBM Quantum (future integration)

**Current status:** not implemented as an execution path. The app only *detects* whether credentials appear
to be configured (`src/ibm_quantum.py`) and shows **"IBM Quantum Hardware — Configuration Required"**
otherwise. It never reports hardware results.

## Secure configuration
Never commit a token. Use either:

```bash
# A) environment variable
export QISKIT_IBM_TOKEN='<token>'

# B) saved account (written to ~/.qiskit/qiskit-ibm.json, outside the repo)
pip install qiskit-ibm-runtime
python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<token>')"
```

`.env`, `.streamlit/secrets.toml` and `qiskit-ibm.json` are in `.gitignore`.

## Planned workflow
1. Local Qiskit simulation (done).
2. Transpile the feature-map circuit for a chosen backend (a generic basis-gate preview exists in the app).
3. Submit fidelity-kernel circuits for a **small** subset of samples (hardware cost grows with N²).
4. Compare hardware kernel entries with the exact simulator values and report the difference, noise included.

Check the current `qiskit-ibm-runtime` documentation for the Sampler/Estimator primitives and account
channels before implementing, since these APIs change between releases.
