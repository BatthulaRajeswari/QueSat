"""Feature extraction: optional spectral indices and band summaries."""
from __future__ import annotations

import pandas as pd

from .data_loader import FEATURES, TARGET

INDEX_COLUMNS = ["NDVI", "NDWI"]

FEATURE_DESCRIPTIONS = {
    "Blue": "Blue band reflectance",
    "Green": "Green band reflectance",
    "Red": "Red band reflectance",
    "NIR": "Near-infrared band reflectance",
    "NDVI": "Vegetation index (NIR − Red) / (NIR + Red)",
    "NDWI": "Water index (Green − NIR) / (Green + NIR)",
}


def add_spectral_indices(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Return a copy of df with NDVI and NDWI columns appended."""
    out = df.copy()
    out["NDVI"] = (out["NIR"] - out["Red"]) / (out["NIR"] + out["Red"] + eps)
    out["NDWI"] = (out["Green"] - out["NIR"]) / (out["Green"] + out["NIR"] + eps)
    return out


def feature_columns(use_indices: bool = False) -> list[str]:
    return FEATURES + (INDEX_COLUMNS if use_indices else [])


def band_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each band for No Change vs Change (computed from the data)."""
    means = df.groupby(TARGET)[FEATURES].mean()
    means.index = ["No Change" if i == 0 else "Change" for i in means.index]
    return means
