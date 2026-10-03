# 🛰️ QueSat — Earth Observation Data Analysis with Quantum ML

**UC-097 · Hybrid Classical–Quantum Earth Observation · Team QueSat**
Rajiv Gandhi University of Knowledge Technologies (RGUKT), Nuzvid, Andhra Pradesh

> **Demonstration Mode — Results generated using synthetic satellite-derived features.**
> The bundled data are synthetic and are not satellite imagery. No quantum advantage is claimed.

## Problem statement
Separating real land-surface change (water bodies, crops, floods, land cover) from noise in multispectral
measurements across large regions is hard to do quickly and repeatably.

## Research gap
Quantum kernels give a different similarity measure for SVMs, yet there are few transparent, application-level,
side-by-side comparisons against a classical baseline on Earth-observation features.

## Proposed solution
A runnable hybrid workflow with an Andhra Pradesh dashboard: classical preprocessing → quantum feature map →
fidelity quantum kernel → QSVM, evaluated against a classical SVM on identical splits, with every number
computed by code.

## Architecture
```
Satellite-derived features → Preprocessing → Feature extraction → Dimensionality reduction
 → RY encoding → Quantum feature map → CNOT entanglement → Quantum kernel → QSVM → Change / No Change
 → Classical SVM comparison → Confusion matrix → Andhra Pradesh regional visualization
```
See [docs/architecture.md](docs/architecture.md).

## Quantum methodology
4 features → 4 qubits. Features (scaled to [0, π]) are encoded with RY rotations, entangled with a linear CNOT
chain (2 layers by default), and the kernel is the state fidelity |⟨φ(x)|φ(y)⟩|². The Gram matrix feeds an SVM
(QSVC-equivalent). Simulated locally with Qiskit (**LOCAL SIMULATION**). Details, including why 2 layers are
the default: [docs/quantum_methodology.md](docs/quantum_methodology.md).

## Classical baseline
scikit-learn `SVC` (RBF) on exactly the same processed features, train set and test set. Metrics: accuracy,
precision, recall, F1, confusion matrix.

## Dataset
* **Synthetic demo data** (offline): Blue, Green, Red, NIR + `Change` (0 = No Change, 1 = Change), one
  generator per scenario. Samples in `data/demo/`.
* **Your own data:** upload `Blue,Green,Red,NIR,Change` (optional `Latitude,Longitude`). Validation covers
  missing columns, NaN, bad labels, small / single-class files.
* Nothing is downloaded automatically.

## Andhra Pradesh focus
Areas: Andhra Pradesh, Krishna, Guntur, Vijayawada Region, Visakhapatnam, Kakinada, Godavari Region, Nellore,
Tirupati — **demonstration Areas of Interest** unless you upload data for them. The regional output is an
*Illustrative Regional Visualization* built from a synthetic scene whose cells are classified by the trained
models. It is never presented as satellite imagery.

## Demo scenarios
Water-body Change · Agricultural Change · Flood-related Change (before → after → detected) · Land-cover Change.
Script: [docs/expo_demo.md](docs/expo_demo.md).

## Installation
Python 3.10+ recommended.
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running
```bash
streamlit run app.py          # dashboard
pytest                         # tests
python scripts/generate_sample_outputs.py   # regenerate sample figures from the demo data
```
On the **Dashboard** pick Area + Scenario and press **RUN ANALYSIS** (or *Expo quick start*).

## Project structure
```
QueSat/
├── app.py                    Streamlit dashboard (8 pages)
├── requirements.txt  pytest.ini  .gitignore  .streamlit/config.toml
├── src/                      data_loader, preprocessing, feature_engineering, classical_model,
│                             quantum_circuit, quantum_model, simulation, evaluation,
│                             visualization, regional_visualization, ibm_quantum
├── data/demo  data/real      synthetic samples / your real CSVs (git-ignored)
├── docs/                     architecture, methodology, quantum_methodology, expo_demo, ibm_quantum,
│                             sample_outputs/
├── scripts/                  generate_sample_outputs.py
├── tests/                    test_data, test_preprocessing, test_classical, test_quantum, reference_sim
└── outputs/
```

## Screenshots
*(Add screenshots of each page here.)* Sample figures: [docs/sample_outputs/](docs/sample_outputs/)
(see its README for provenance).

## IBM Quantum (future)
Not executed on hardware. The IBM page reports **"IBM Quantum Hardware — Configuration Required"** until
credentials exist, and never shows hardware results. Keys go in environment variables / a saved account,
never in code: [docs/ibm_quantum.md](docs/ibm_quantum.md).

## Limitations
* Synthetic data demonstrate the pipeline, not the condition of any real region.
* Quantum results are classical simulation; simulation time is not hardware speed.
* Single train/test split; no cross-validation or noise models yet.
* Fidelity kernels scale as O(N²) in circuit evaluations; large uploads are stratified-subsampled (default 500).
* The Streamlit UI and Qiskit code were written and checked in an environment without Qiskit/Streamlit;
  run `pytest` and the app once after installing requirements (see tests marked *skipped* without Qiskit).

## Future scope
Sentinel-2 derived features, temporal series, coordinates and region metadata, NDVI-style indices at scale,
repeated cross-validation, noise-aware simulation, IBM Quantum hardware runs, other kernels / feature maps.
