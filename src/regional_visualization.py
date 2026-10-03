"""Regional visualization.

Without real geographic data the map is an ILLUSTRATIVE SYNTHETIC SCENE:
  1. a procedural landscape (before / after) defines which cells changed;
  2. synthetic band values are drawn for every cell from the scenario generator;
  3. the TRAINED models classify every cell — the "Detected change" panel is the actual
     model output, not the scene definition.
The scene is never satellite imagery and is always labelled as illustrative.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

from .data_loader import FEATURES, SYNTHETIC_TAG, sample_bands
from .feature_engineering import add_spectral_indices

SCENE_SIZE = 48
CHANGE_COLOR = "#F97316"
NOCHANGE_COLOR = "#9FD8D0"

CATEGORIES = {
    "Water-body Change": {0: ("Land", "#CDB98F"), 1: ("Water", "#2F6FED")},
    "Agricultural Change": {0: ("Fallow / bare", "#D9C79E"), 1: ("Cropland", "#5BAE5B")},
    "Flood-related Change": {0: ("Dry land", "#CDB98F"), 1: ("Water", "#2F6FED")},
    "Land-cover Change": {0: ("Vegetation", "#4C9A5A"), 1: ("Built-up", "#8A8F9C"), 2: ("Bare land", "#D9C79E")},
}


def _noise(rng, size, sigma):
    f = gaussian_filter(rng.normal(size=(size, size)), sigma, mode="reflect")
    return (f - f.min()) / (f.max() - f.min() + 1e-12)


def _river(rng, size):
    yy, xx = np.mgrid[0:size, 0:size]
    y0 = size * rng.uniform(0.35, 0.65)
    amp = size * rng.uniform(0.08, 0.16)
    freq = rng.uniform(1.0, 1.8)
    phase = rng.uniform(0, 2 * np.pi)
    centre = y0 + amp * np.sin(2 * np.pi * freq * xx / size + phase)
    return yy, xx, np.abs(yy - centre)


def build_scene(region: str, scenario: str, size: int = SCENE_SIZE) -> dict:
    """Procedural before/after landscape. Layout depends on region + scenario (deterministic)."""
    if scenario not in CATEGORIES:
        raise ValueError(f"Unknown scenario '{scenario}'.")
    rng = np.random.default_rng(zlib.crc32(f"scene|{region}|{scenario}".encode()))

    if scenario == "Water-body Change":
        yy, xx, dist = _river(rng, size)
        n = _noise(rng, size, 3.0)
        before = ((dist < 1.6) | (n > 0.82)).astype(int)
        after = ((dist < 1.6 + 2.2 * (0.4 + 0.6 * xx / size)) | (n > 0.74)).astype(int)

    elif scenario == "Agricultural Change":
        b = 6
        wy = (_noise(rng, size, 4.0) * b * 1.2).astype(int)
        wx = (_noise(rng, size, 4.0) * b * 1.2).astype(int)
        yy, xx = np.mgrid[0:size, 0:size]
        cols = size // b + 4
        pid = ((yy + wy) // b) * cols + ((xx + wx) // b)
        table_crop = rng.random(cols * cols + 10)
        table_flip = rng.random(cols * cols + 10)
        before = (table_crop[pid] < 0.55).astype(int)
        flip = table_flip[pid] < 0.28
        after = np.where(flip, 1 - before, before)

    elif scenario == "Flood-related Change":
        yy, xx, dist = _river(rng, size)
        elev = 0.6 * _noise(rng, size, 6.0) + 0.4 * np.clip(dist / (size * 0.5), 0, 1)
        elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-12)
        before = (dist < 1.6).astype(int)
        after = (before.astype(bool) | (elev < 0.28)).astype(int)

    else:  # Land-cover Change
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = size * rng.uniform(0.3, 0.7), size * rng.uniform(0.3, 0.7)
        d = np.hypot(xx - cx, yy - cy)
        n1, n2 = _noise(rng, size, 5.0), _noise(rng, size, 3.0)
        before = np.zeros((size, size), dtype=int)
        before[n1 < 0.25] = 2
        before[d < 5] = 1
        after = before.copy()
        after[(d < 10) & (before != 1)] = 1
        after[(n2 > 0.80) & (before == 0) & (after == 0)] = 2

    truth = (before != after).astype(int)
    return {"region": region, "scenario": scenario, "size": size,
            "before": before, "after": after, "truth": truth,
            "categories": CATEGORIES[scenario]}


def classify_scene(scene: dict, prepared, feature_cols: list, use_indices: bool, predictors: dict):
    """Classify every cell with the trained models.

    predictors: {model name: callable(X_final_features) -> 0/1 array}
    Returns (predictions dict, errors dict).
    """
    shape = scene["truth"].shape
    truth = scene["truth"].ravel()
    rng = np.random.default_rng(zlib.crc32(f"cells|{scene['region']}|{scene['scenario']}".encode()))
    bands = sample_bands(scene["scenario"], truth, rng, scene["region"])
    cells = pd.DataFrame(bands, columns=FEATURES)
    if use_indices:
        cells = add_spectral_indices(cells)
    X = prepared.transform(cells[feature_cols].to_numpy(dtype=float))
    preds, errors = {}, {}
    for name, fn in predictors.items():
        try:
            preds[name] = np.asarray(fn(X)).astype(int).reshape(shape)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return preds, errors


def summarize_scene(scene: dict, pred: np.ndarray) -> dict:
    truth = scene["truth"]
    return {
        "cells": int(truth.size),
        "detected_pct": float(pred.mean() * 100),
        "scene_change_pct": float(truth.mean() * 100),
        "cell_accuracy_pct": float((pred == truth).mean() * 100),
        "detected_cells": int(pred.sum()),
    }


def regional_map_fig(scene: dict, pred: np.ndarray, model_name: str) -> Figure:
    """Before / After / Detected-change panels for the illustrative scene."""
    cats = scene["categories"]
    cmap = ListedColormap([cats[k][1] for k in sorted(cats)])
    vmax = max(len(cats) - 1, 1)
    fig = Figure(figsize=(13, 5.4), facecolor="white")
    axes = fig.subplots(1, 3)

    axes[0].imshow(scene["before"], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    axes[0].set_title("Before (illustrative)", fontsize=11, color="#0B1F3A")
    axes[1].imshow(scene["after"], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    axes[1].set_title("After (illustrative)", fontsize=11, color="#0B1F3A")
    axes[2].imshow(pred, cmap=ListedColormap([NOCHANGE_COLOR, CHANGE_COLOR]), vmin=0, vmax=1,
                   interpolation="nearest")
    axes[2].set_title(f"Detected change — {model_name}", fontsize=11, color="#0B1F3A")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#0B1F3A")
            s.set_linewidth(1.2)
    axes[2].annotate("", xy=(0.92, 0.95), xytext=(0.92, 0.80), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="#0B1F3A", lw=1.5))
    axes[2].text(0.92, 0.97, "N", transform=axes[2].transAxes, ha="center", va="bottom",
                 fontsize=9, color="#0B1F3A", fontweight="bold")

    fig.suptitle(f"{scene['region']}  ·  {scene['scenario']}", fontsize=14, fontweight="bold",
                 color="#0B1F3A", y=0.98)
    cat_handles = [Patch(color=c, label=n) for _, (n, c) in sorted(cats.items())]
    det_handles = [Patch(color=CHANGE_COLOR, label="Change"), Patch(color=NOCHANGE_COLOR, label="No Change")]
    fig.legend(handles=cat_handles, loc="lower left", bbox_to_anchor=(0.04, 0.06), ncol=len(cat_handles),
               frameon=False, fontsize=9, title="Land state (illustrative)", title_fontsize=9)
    fig.legend(handles=det_handles, loc="lower right", bbox_to_anchor=(0.97, 0.06), ncol=2,
               frameon=False, fontsize=9, title="Model classification", title_fontsize=9)
    fig.text(0.5, 0.012, f"{SYNTHETIC_TAG} · Illustrative regional visualization, "
             "not satellite imagery, not to scale.", ha="center", fontsize=9, color="#B42318")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.22, wspace=0.06)
    return fig


def geo_scatter_fig(df: pd.DataFrame, pred: np.ndarray, model_name: str) -> Figure:
    """Scatter of real supplied coordinates coloured by model prediction (no basemap tiles)."""
    fig = Figure(figsize=(7.5, 5.5), facecolor="white")
    ax = fig.subplots()
    mask = np.asarray(pred).astype(bool)
    ax.scatter(df.loc[~mask, "Longitude"], df.loc[~mask, "Latitude"], s=18, c=NOCHANGE_COLOR, label="No Change")
    ax.scatter(df.loc[mask, "Longitude"], df.loc[mask, "Latitude"], s=18, c=CHANGE_COLOR, label="Change")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Predicted change at supplied coordinates — {model_name}", color="#0B1F3A")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.01, "Coordinates from the uploaded dataset. No basemap or imagery is downloaded.",
             ha="center", fontsize=8, color="#555")
    fig.subplots_adjust(bottom=0.14)
    return fig
