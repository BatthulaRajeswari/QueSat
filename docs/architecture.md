# Architecture

```
app.py (Streamlit UI, 8 pages)
   │  session_state: selections, config, last result
   ▼
src/simulation.run_pipeline()  ── orchestrates, isolates failures per model
   ├─ data_loader.py            synthetic demo generator, CSV validation (never raises)
   ├─ feature_engineering.py    optional NDVI / NDWI, band summaries
   ├─ preprocessing.py          stratified split → StandardScaler → PCA (only if needed) → [0, π] angles
   ├─ classical_model.py        scikit-learn SVC (RBF) on the shared features
   ├─ quantum_circuit.py        Qiskit feature map, exact statevectors, fidelity Gram matrix
   ├─ quantum_model.py          QuantumKernelSVM (precomputed-kernel SVC) + run_quantum()
   ├─ evaluation.py             accuracy / precision / recall / F1 / confusion matrix, neutral summaries
   ├─ regional_visualization.py illustrative before/after scene, per-cell classification, map figures
   ├─ visualization.py          matplotlib figures (kernel heatmap, comparison, feature space …)
   └─ ibm_quantum.py            credential *detection* only (no hardware execution)
```

## Data flow

1. Dataset (synthetic demo or validated upload) → optional spectral indices.
2. Stratified train/test split; scaler / PCA / angle scaler are fitted on **train only**.
3. The same `X_train`, `X_test` feed both the classical SVM and the quantum-kernel SVM.
4. Metrics are computed from real predictions on the test split.
5. If data are synthetic, a procedural scene is classified cell-by-cell by the trained models.
   If an upload has `Latitude`/`Longitude`, a coordinate scatter is drawn instead.

## Failure isolation

Each model runs in its own `try` block. If Qiskit is missing or the quantum kernel fails, the result
carries `quantum.status == "error"` with a user-facing message; the classical result, charts and
comparison remain available. Nothing substitutes fake numbers.

## Extending to real data

Add a loader that outputs `Blue, Green, Red, NIR, Change` (+ optional `Latitude`, `Longitude`, and further
bands). `feature_engineering.py` is the place for new indices; `run_pipeline` takes the feature list from
`feature_columns()`. No data are downloaded automatically.
