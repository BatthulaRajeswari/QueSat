"""Data loading, synthetic demo generation and CSV validation.

The demo generator produces SYNTHETIC satellite-derived features. It is NOT real
satellite imagery and must always be labelled as such in any user interface.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Blue", "Green", "Red", "NIR"]
TARGET = "Change"
REQUIRED = FEATURES + [TARGET]
GEO_COLUMNS = ["Latitude", "Longitude"]

MIN_SAMPLES = 30
MIN_PER_CLASS = 5

DEMO_BANNER = ("Demonstration Mode — Results generated using synthetic "
               "satellite-derived features.")
SYNTHETIC_TAG = "Illustrative Demonstration — Synthetic Data"

AP_REGION = "Andhra Pradesh"
AREAS = [
    "Andhra Pradesh", "Krishna", "Guntur", "Vijayawada Region",
    "Visakhapatnam", "Kakinada", "Godavari Region", "Nellore", "Tirupati",
]

# Rough bounding box of Andhra Pradesh (used only for a plausibility note on uploads).
AP_BBOX = {"lat": (12.6, 19.95), "lon": (76.7, 84.8)}

# Class-conditional mean band values (Blue, Green, Red, NIR) of the synthetic generator.
SCENARIOS: dict[str, dict] = {
    "Water-body Change": {
        "description": "Land converting to open water (reservoir / river widening / new ponds).",
        "nochange": [0.30, 0.40, 0.34, 0.58],
        "change": [0.37, 0.46, 0.28, 0.34],
        "sd": 0.085,
        "signature": "Synthetic generator: NIR drops strongly, Blue/Green rise slightly where water appears.",
    },
    "Agricultural Change": {
        "description": "Cropland changing state (harvest / fallow / crop-cycle change).",
        "nochange": [0.20, 0.35, 0.25, 0.62],
        "change": [0.26, 0.38, 0.37, 0.45],
        "sd": 0.085,
        "signature": "Synthetic generator: NIR falls and Red rises where vegetation vigour is lost.",
    },
    "Flood-related Change": {
        "description": "Dry land inundated by flood water (before → after).",
        "nochange": [0.28, 0.38, 0.36, 0.55],
        "change": [0.38, 0.45, 0.33, 0.28],
        "sd": 0.09,
        "signature": "Synthetic generator: very low NIR with raised Blue/Green over inundated land.",
    },
    "Land-cover Change": {
        "description": "Vegetation converting to built-up or bare land.",
        "nochange": [0.18, 0.32, 0.22, 0.60],
        "change": [0.30, 0.34, 0.38, 0.40],
        "sd": 0.085,
        "signature": "Synthetic generator: NIR falls while Red and Blue rise over built-up / bare surfaces.",
    },
}


def scenario_slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def region_offset(region: str) -> np.ndarray:
    """Small deterministic per-region band offset so demo regions differ slightly."""
    rng = np.random.default_rng(zlib.crc32(f"offset|{region}".encode()))
    return rng.normal(0.0, 0.015, size=4)


def sample_bands(scenario: str, labels: np.ndarray, rng: np.random.Generator,
                 region: str = AP_REGION) -> np.ndarray:
    """Draw synthetic Blue/Green/Red/NIR values for the given 0/1 labels."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}.")
    spec = SCENARIOS[scenario]
    labels = np.asarray(labels).astype(int).reshape(-1)
    mu = np.where(labels[:, None] == 1, np.array(spec["change"]), np.array(spec["nochange"]))
    noise = rng.normal(0.0, spec["sd"], size=(len(labels), 4))
    brightness = rng.normal(0.0, spec["sd"] * 0.6, size=(len(labels), 1))  # shared illumination term
    x = mu + region_offset(region) + noise + brightness
    return np.clip(x, 0.01, 0.99)


def generate_demo_data(region: str = AP_REGION, scenario: str = "Water-body Change",
                       n: int = 240, change_fraction: float = 0.45,
                       seed: int | None = None) -> pd.DataFrame:
    """Create a SYNTHETIC demo dataset (deterministic for a given region/scenario/seed)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(SCENARIOS)}.")
    if seed is None:
        seed = zlib.crc32(f"{region}|{scenario}".encode())
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < change_fraction).astype(int)
    # guarantee both classes are present
    labels[0], labels[1] = 0, 1
    x = sample_bands(scenario, labels, rng, region)
    df = pd.DataFrame(x.round(4), columns=FEATURES)
    df[TARGET] = labels
    return df


# --------------------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None
    stats: dict = field(default_factory=dict)


def _column_mapping(columns) -> dict:
    lookup = {c.lower(): c for c in REQUIRED}
    lookup.update({"latitude": "Latitude", "lat": "Latitude",
                   "longitude": "Longitude", "lon": "Longitude", "lng": "Longitude"})
    mapping = {}
    for c in columns:
        key = str(c).strip().lower()
        if key in lookup:
            mapping[c] = lookup[key]
    return mapping


def validate_dataframe(df: pd.DataFrame | None) -> ValidationReport:
    """Validate a user-supplied dataframe. Never raises; reports problems instead."""
    errors: list[str] = []
    warnings: list[str] = []
    if df is None or len(df) == 0:
        return ValidationReport(False, ["The file is empty. Expected header: Blue,Green,Red,NIR,Change"])

    mapping = _column_mapping(df.columns)
    df = df.rename(columns=mapping)
    if df.columns.duplicated().any():
        return ValidationReport(False, ["Duplicate column names were found after normalising the header."])

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return ValidationReport(False, [
            f"Missing required column(s): {', '.join(missing)}. "
            "Expected header: Blue,Green,Red,NIR,Change"])

    geo_present = all(c in df.columns for c in GEO_COLUMNS)
    keep = REQUIRED + (GEO_COLUMNS if geo_present else [])
    work = df[keep].copy()
    n_raw = len(work)
    for c in keep:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)

    missing_cells = int(work[REQUIRED].isna().sum().sum())
    bad_rows = work[REQUIRED].isna().any(axis=1)
    n_bad = int(bad_rows.sum())
    if n_bad:
        warnings.append(f"{n_bad} row(s) with missing / non-numeric / infinite values were dropped.")
    work = work.loc[~bad_rows].reset_index(drop=True)

    if len(work):
        bad_labels = ~work[TARGET].isin([0, 1])
        if bad_labels.any():
            vals = sorted(work.loc[bad_labels, TARGET].unique().tolist())[:5]
            errors.append("The 'Change' column must contain only 0 (No Change) and 1 (Change); "
                          f"found other values such as {vals}.")
    if not errors:
        work[TARGET] = work[TARGET].astype(int)
        if len(work) < MIN_SAMPLES:
            errors.append(f"Dataset too small: {len(work)} valid rows (minimum {MIN_SAMPLES}).")
        else:
            counts = work[TARGET].value_counts()
            for cls, label in ((0, "No Change"), (1, "Change")):
                if int(counts.get(cls, 0)) < MIN_PER_CLASS:
                    errors.append(f"Class '{label}' has fewer than {MIN_PER_CLASS} samples; "
                                  "both classes are needed for classification.")
    if not errors:
        feats = work[FEATURES]
        if (feats < 0).any().any() or (feats > 1).any().any():
            warnings.append("Some band values lie outside 0–1. This is acceptable for digital numbers or "
                            "scaled data; StandardScaler normalises the features.")
        if geo_present and work[GEO_COLUMNS].isna().any().any():
            warnings.append("Latitude/Longitude contain gaps; the coordinate map will be unavailable.")

    stats = {
        "n_raw": n_raw,
        "n_valid": int(len(work)),
        "n_features": len(FEATURES),
        "class_counts": {int(k): int(v) for k, v in work[TARGET].value_counts().items()} if len(work) else {},
        "missing_cells": missing_cells,
        "dropped_rows": n_bad,
        "has_geo": bool(geo_present),
    }
    return ValidationReport(not errors, errors, warnings, work if not errors else None, stats)


def load_csv(file_or_path) -> ValidationReport:
    """Read and validate a CSV file (path or file-like object)."""
    try:
        df = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        return ValidationReport(False, ["The file is empty or has no header row."])
    except UnicodeDecodeError:
        return ValidationReport(False, ["The file could not be decoded as UTF-8 text. Please upload a plain CSV."])
    except Exception as exc:  # parser errors etc.
        return ValidationReport(False, [f"The file could not be read as CSV ({type(exc).__name__}: {exc})."])
    return validate_dataframe(df)


def has_geo(df: pd.DataFrame | None) -> bool:
    return (df is not None and all(c in df.columns for c in GEO_COLUMNS)
            and not df[GEO_COLUMNS].isna().any().any())


def fraction_inside_ap(df: pd.DataFrame) -> float:
    """Fraction of points inside the rough Andhra Pradesh bounding box."""
    lat, lon = df["Latitude"], df["Longitude"]
    inside = lat.between(*AP_BBOX["lat"]) & lon.between(*AP_BBOX["lon"])
    return float(inside.mean())


def write_demo_csvs(out_dir: Path | None = None, region: str = "Krishna") -> list[Path]:
    """Write one synthetic CSV per scenario (used to ship sample files in data/demo)."""
    out_dir = out_dir or Path(__file__).resolve().parents[1] / "data" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario in SCENARIOS:
        p = out_dir / f"demo_{scenario_slug(scenario)}.csv"
        generate_demo_data(region, scenario).to_csv(p, index=False)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_demo_csvs():
        print("wrote", p)
