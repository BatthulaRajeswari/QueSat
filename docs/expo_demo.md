# Expo demo script (3–5 minutes)

Start: `streamlit run app.py`. A purple **Demo Mode · Synthetic Data** pill is visible whenever synthetic
data are in use. Say this out loud: *the data are synthetic; the pipeline and measurements are real.*

| # | Action | Page | What to say |
|---|---|---|---|
| 1 | Region of Interest = Andhra Pradesh | Dashboard | Demonstration Areas of Interest. |
| 2 | Area = Krishna | Dashboard | Selectable AP areas. |
| 3 | Scenario = Water-body Change | Dashboard | Four scenarios. |
| 4 | **RUN ANALYSIS** (or *Expo quick start*) | Dashboard | Live cards: samples, features, qubits, status. |
| 5 | Show band data | Data Analysis | Blue/Green/Red/NIR, validation, class balance. |
| 6 | Show preprocessing | Data Analysis | StandardScaler, PCA only if needed, 4 features → 4 qubits. |
| 7 | Show the circuit | Quantum Circuit | RY encoding + CNOT entanglement; kernel = state overlap. |
| 8 | Show kernel heatmap | Quantum Model | Block structure = similarity within classes. |
| 9 | Show QSVM output | Quantum Model | Predictions and decision values (not probabilities). |
| 10 | Compare models | Classical vs Quantum | Same data, same split. Report numbers; do not claim a winner. |
| 11 | Regional output | Results & Visualization | Illustrative map: before / after / **model-detected** change. |
| 12 | Future work | IBM Quantum | Honest status: configuration required, no hardware used. |

Other scenarios: Guntur · Agricultural, Godavari Region · Flood-related (before → after → detected),
Vijayawada Region · Land-cover.

## Likely questions
* *"Is this real satellite data?"* No — synthetic, labelled everywhere. Real CSVs can be uploaded.
* *"Is quantum better?"* The page shows measured numbers on one split; no general claim is made.
* *"Does it run on a quantum computer?"* No — local simulation. IBM hardware is future work.
