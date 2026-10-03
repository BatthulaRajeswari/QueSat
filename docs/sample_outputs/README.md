# Sample outputs — provenance

These figures were generated from the **synthetic demo data** with `scripts/generate_sample_outputs.py`.

**Important:** they carry the suffix `_reference-sim` because the build environment could not install Qiskit.
The quantum-kernel values (kernel heatmaps, QSVM metrics, QSVM map panels) were therefore computed with the
independent NumPy simulator in `tests/reference_sim.py`, which implements the *same* circuit
(RY encoding + linear CNOT, 2 layers). They are **not** Qiskit outputs, and the numbers are from a single
split of synthetic data — they do not show that either model is better.

To regenerate with Qiskit (writes files without the suffix, and uses the real pipeline):

```bash
pip install -r requirements.txt
python scripts/generate_sample_outputs.py
```

Files: `*_map.png` (illustrative regional visualization), `*_comparison.png`, `*_kernel.png`,
`*_feature_space.png`, `metrics_reference-sim.json`.
