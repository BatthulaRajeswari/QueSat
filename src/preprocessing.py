"""Preprocessing: split, StandardScaler, optional PCA, and RY-angle scaling.

All fitting happens on the TRAINING split only. The final representation
(`X_train`, `X_test`) is shared by the classical SVM and the quantum kernel SVM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data_loader import TARGET


@dataclass
class PreparedData:
    feature_names: list
    component_names: list
    X_train: np.ndarray            # final features, angles in [0, pi]
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_scaled: np.ndarray     # after StandardScaler (+PCA), before angle scaling
    X_test_scaled: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray
    scaler: StandardScaler
    pca: PCA | None
    angle_scaler: MinMaxScaler
    n_input_features: int
    n_components: int
    used_pca: bool
    explained_variance: list | None
    notes: list

    def transform(self, X_raw) -> np.ndarray:
        """Map raw feature rows (same columns as feature_names) to the final representation."""
        z = self.scaler.transform(np.asarray(X_raw, dtype=float))
        if self.pca is not None:
            z = self.pca.transform(z)
        return np.clip(self.angle_scaler.transform(z), 0.0, np.pi)


def preprocess(df: pd.DataFrame, feature_cols: list, n_components: int | None = None,
               test_size: float = 0.25, seed: int = 42) -> PreparedData:
    if len(df) < 4:
        raise ValueError("Not enough samples to split into train and test sets.")
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)
    if np.isnan(X).any():
        raise ValueError("Features contain NaN values; validate the data before preprocessing.")

    notes = []
    n_in = len(feature_cols)
    if n_components is None:
        n_components = n_in
    requested = n_components
    n_components = int(min(max(1, n_components), n_in))
    if n_components != requested:
        notes.append(f"Requested feature dimension {requested} was adjusted to {n_components} "
                     f"(available input features: {n_in}).")

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X[idx_tr])
    z_tr, z_te = scaler.transform(X[idx_tr]), scaler.transform(X[idx_te])

    pca, explained = None, None
    used_pca = n_components < n_in
    if used_pca:
        pca = PCA(n_components=n_components, random_state=seed).fit(z_tr)
        z_tr, z_te = pca.transform(z_tr), pca.transform(z_te)
        explained = [float(v) for v in pca.explained_variance_ratio_]
        names = [f"PC{i + 1}" for i in range(n_components)]
    else:
        names = list(feature_cols)
        notes.append("PCA not required: feature dimension equals the number of input features.")

    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(z_tr)
    a_tr = np.clip(angle_scaler.transform(z_tr), 0.0, np.pi)
    a_te = np.clip(angle_scaler.transform(z_te), 0.0, np.pi)
    notes.append("Features are rescaled to [0, π] (fit on training data, test values clipped) "
                 "so they can be used as RY rotation angles.")

    return PreparedData(
        feature_names=list(feature_cols), component_names=names,
        X_train=a_tr, X_test=a_te, y_train=y[idx_tr], y_test=y[idx_te],
        X_train_scaled=z_tr, X_test_scaled=z_te, idx_train=idx_tr, idx_test=idx_te,
        scaler=scaler, pca=pca, angle_scaler=angle_scaler,
        n_input_features=n_in, n_components=n_components, used_pca=used_pca,
        explained_variance=explained, notes=notes)
