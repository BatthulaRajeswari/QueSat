"""Charts built from actual results. All functions return matplotlib Figure objects."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from .data_loader import FEATURES, TARGET
from .evaluation import CLASSICAL_NAME, QUANTUM_NAME

NAVY, BLUE, TEAL, PURPLE = "#0B1F3A", "#2F6FED", "#14B8A6", "#7C3AED"
CHANGE, NOCHANGE = "#F97316", "#14B8A6"
KERNEL_CMAP = LinearSegmentedColormap.from_list("quesat", ["#0B1F3A", "#2F6FED", "#14B8A6", "#F4FBFA"])


def _fig(w=6.5, h=4.2):
    fig = Figure(figsize=(w, h), facecolor="white")
    return fig, fig.subplots()


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=NAVY, labelsize=9)
    ax.title.set_color(NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)


def class_distribution_fig(df) -> Figure:
    fig, ax = _fig(4.6, 3.4)
    counts = df[TARGET].value_counts().reindex([0, 1]).fillna(0).astype(int)
    bars = ax.bar(["No Change", "Change"], counts.values, color=[NOCHANGE, CHANGE], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10, color=NAVY)
    ax.set_title("Class distribution")
    ax.set_ylabel("Samples")
    _style(ax)
    fig.tight_layout()
    return fig


def feature_distribution_fig(df) -> Figure:
    fig = Figure(figsize=(10, 3.2), facecolor="white")
    axes = fig.subplots(1, len(FEATURES))
    for ax, f in zip(axes, FEATURES):
        for label, color, name in ((0, NOCHANGE, "No Change"), (1, CHANGE, "Change")):
            ax.hist(df.loc[df[TARGET] == label, f], bins=18, alpha=0.65, color=color, label=name)
        ax.set_title(f)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def band_means_fig(summary) -> Figure:
    """Grouped bars: mean band value for No Change vs Change (computed from the data)."""
    fig, ax = _fig(6.5, 3.6)
    x = np.arange(len(FEATURES))
    w = 0.36
    ax.bar(x - w / 2, summary.loc["No Change", FEATURES].values, w, color=NOCHANGE, label="No Change")
    ax.bar(x + w / 2, summary.loc["Change", FEATURES].values, w, color=CHANGE, label="Change")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES)
    ax.set_ylabel("Mean band value")
    ax.set_title("Satellite-derived feature analysis (mean per class)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def kernel_heatmap_fig(K, y, title="Quantum kernel matrix (training set, sorted by class)") -> Figure:
    order = np.argsort(np.asarray(y), kind="stable")
    Ks = np.asarray(K)[np.ix_(order, order)]
    fig, ax = _fig(5.8, 5.0)
    im = ax.imshow(Ks, cmap=KERNEL_CMAP, vmin=0, vmax=1)
    n0 = int((np.asarray(y) == 0).sum())
    ax.axhline(n0 - 0.5, color="white", lw=1)
    ax.axvline(n0 - 0.5, color="white", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Training sample (No Change | Change)")
    ax.set_ylabel("Training sample")
    fig.colorbar(im, ax=ax, label="Fidelity |⟨φ(x)|φ(y)⟩|²", fraction=0.046)
    _style(ax)
    fig.tight_layout()
    return fig


def comparison_chart_fig(table) -> Figure:
    """Grouped bars of measured metrics per model. Unavailable models are omitted and noted."""
    fig, ax = _fig(7.5, 4.2)
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    avail = table[table["Status"] == "ok"]
    colors = {CLASSICAL_NAME: BLUE, QUANTUM_NAME: PURPLE}
    n = max(len(avail), 1)
    w = 0.8 / n
    x = np.arange(len(metrics))
    for i, (_, row) in enumerate(avail.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=row["Model"], color=colors.get(row["Model"], TEAL))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=NAVY)
    missing = table[table["Status"] != "ok"]["Model"].tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_title("Experimental Performance Comparison")
    if missing:
        ax.text(0.5, 0.5, "Unavailable: " + ", ".join(missing), transform=ax.transAxes, ha="center",
                fontsize=9, color="#B42318")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    _style(ax)
    fig.tight_layout()
    return fig


def confusion_matrix_fig(cm, title) -> Figure:
    cm = np.asarray(cm)
    fig, ax = _fig(4.2, 3.6)
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("cm", ["#F3F6FB", BLUE]))
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.55
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center", fontsize=12,
                    color="white" if dark else NAVY)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Change", "Change"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Change", "Change"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    _style(ax)
    fig.tight_layout()
    return fig


def feature_space_fig(prepared) -> Figure:
    """2-D PCA of the standardised features, coloured by class (train + test)."""
    from numpy import vstack, concatenate
    Z = vstack([prepared.X_train_scaled, prepared.X_test_scaled])
    y = concatenate([prepared.y_train, prepared.y_test])
    is_test = np.r_[np.zeros(len(prepared.y_train), bool), np.ones(len(prepared.y_test), bool)]
    if Z.shape[1] >= 2:
        P = PCA(n_components=2).fit_transform(Z)
        xl, yl = "Principal component 1", "Principal component 2"
    else:
        P = np.c_[Z[:, 0], np.zeros(len(Z))]
        xl, yl = "Feature 1", ""
    fig, ax = _fig(6.2, 4.6)
    for label, color, name in ((0, NOCHANGE, "No Change"), (1, CHANGE, "Change")):
        m = (y == label) & ~is_test
        ax.scatter(P[m, 0], P[m, 1], s=22, c=color, alpha=0.75, label=f"{name} (train)")
        m = (y == label) & is_test
        ax.scatter(P[m, 0], P[m, 1], s=40, c=color, edgecolors=NAVY, linewidths=1.0, label=f"{name} (test)")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title("Compact Satellite Feature Space")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    _style(ax)
    fig.tight_layout()
    return fig


def decision_hist_fig(decision, y_true, title) -> Figure:
    fig, ax = _fig(5.6, 3.6)
    decision, y_true = np.asarray(decision), np.asarray(y_true)
    for label, color, name in ((0, NOCHANGE, "No Change"), (1, CHANGE, "Change")):
        ax.hist(decision[y_true == label], bins=15, alpha=0.7, color=color, label=f"Actual: {name}")
    ax.axvline(0, color=NAVY, ls="--", lw=1)
    ax.set_xlabel("SVM decision value (> 0 → predicted Change)")
    ax.set_ylabel("Test samples")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def counts_fig(counts: dict, title="Aer shot counts (local simulation)") -> Figure:
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:16]
    fig, ax = _fig(7.0, 3.4)
    ax.bar([k for k, _ in items], [v for _, v in items], color=BLUE)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Measured bitstring")
    ax.set_ylabel("Shots")
    ax.tick_params(axis="x", rotation=60)
    _style(ax)
    fig.tight_layout()
    return fig


def pca_variance_fig(explained) -> Figure:
    fig, ax = _fig(4.6, 3.2)
    ax.bar([f"PC{i + 1}" for i in range(len(explained))], explained, color=PURPLE)
    ax.set_ylim(0, 1)
    ax.set_title("PCA explained variance ratio")
    _style(ax)
    fig.tight_layout()
    return fig
