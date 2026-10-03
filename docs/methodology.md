# Methodology

## Task
Binary classification of satellite-derived feature vectors into **0 = No Change** and **1 = Change**.

## Data
* **Synthetic demo dataset** (`src/data_loader.py`): Blue, Green, Red, NIR values drawn from class-conditional
  Gaussians (one parameter set per scenario) with a shared "illumination" term and a small per-region offset.
  The class means are chosen to mimic qualitative spectral behaviour (e.g. NIR drops under water). It is
  **not** satellite imagery and carries no real-world information about any region.
* **Uploaded CSV**: `Blue,Green,Red,NIR,Change` (+ optional `Latitude`,`Longitude`). Validation checks columns
  (case-insensitive), numeric values, NaN/inf (rows dropped with a warning), labels ∈ {0,1}, minimum 30 rows
  and ≥ 5 samples per class.

## Preprocessing
1. Stratified split (default 75 / 25, seed 42).
2. `StandardScaler` fitted on the training split.
3. `PCA` only when the requested feature dimension is smaller than the number of input features.
4. `MinMaxScaler` to [0, π] (fitted on train, test clipped) so values are valid RY angles.

## Models (identical inputs)
* **Classical:** `sklearn.svm.SVC(kernel="rbf", C=1, gamma="scale")`.
* **Quantum:** fidelity quantum kernel + `SVC(kernel="precomputed", C=1)` — see `quantum_methodology.md`.

## Evaluation
Accuracy, precision, recall, F1 and the confusion matrix on the held-out test split. Reported as measured.
The dashboard states a numeric difference but never declares a winner. One split on one dataset cannot
support a general claim; repeated splits / cross-validation are listed as future work.

## Regional visualization (illustrative)
A procedural before/after landscape defines which grid cells changed. Per-cell band values are drawn from the
same synthetic generator, then **classified by the trained models**. The map's "Detected change" panel is
therefore real model output on synthetic inputs, and is always labelled
*Illustrative Demonstration — Synthetic Data*.
