"""The two figures, both built from the same panel: one source per column.

Each panel carries the individual snapshots of that source (thin), their mean
over the whole block (thick), and the truth mean of the SAME quantity (dashed
black), so a model is read against its reference without leaving its panel.
Axes are shared, otherwise each panel would be auto-stretched and they would all
look alike by construction; the same snapshots are drawn in every panel and in
every figure.
"""

from __future__ import annotations

import zlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fields as F
from utils import FIG_DPI, save_figure

# Hue = model family, so a multi-resolution model wears its clean twin's colour
# and the eye can follow the same curve from one figure to the next. The truth
# is not a model among others: it carries the ink of the text.
COLOR = {"truth": "#0b0b0b", "pca": "#2a78d6", "ae": "#eb6834", "tbd": "#1baf7a"}
# Hues kept for the runs named on the command line, which have no family of their
# own; the choice is a hash of the name, so a run keeps its colour across figures.
EXTRA = ("#8e5bd0", "#d63b6a", "#c79a12", "#3aa8c1")
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d7d2"
UNIT_G, UNIT_I = "W²·m⁻⁴", "W²·m⁻⁴·sr⁻²"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": FIG_DPI, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def color(key):
    """Colour of a source: its family hue, or a stable extra one for a run."""
    if key in COLOR:
        return COLOR[key]
    return EXTRA[zlib.crc32(key.encode()) % len(EXTRA)]


def sample_idx(n_tot, n_show, seed=0):
    """The snapshots drawn as thin curves -- the same ones in every figure."""
    return np.random.default_rng(seed).choice(n_tot, size=min(n_show, n_tot),
                                              replace=False)


def ylim(arrays):
    """One vertical range for a whole family of figures, headroom for the legend."""
    a = list(arrays)
    lo = max(1e-2, min(np.nanpercentile(x, 1) for x in a))
    hi = max(np.nanpercentile(x, 99.9) for x in a) * 3
    return float(lo), float(hi)


def _panels(curves, keys, k, xscale, ylims, unit, idx, out, suptitle=None):
    """curves[key]: (n_snapshots, nbins) for one source."""
    n_tot = curves[keys[0]].shape[0]
    truth_mean = curves["truth"].mean(axis=0) if "truth" in curves else None

    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 4.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for j in idx:
            ax.plot(k, curves[key][j], color=color(key), lw=0.4, alpha=0.13,
                    zorder=1)
        if truth_mean is not None:
            ax.plot(k, truth_mean, color=INK, lw=1.3, ls="--", zorder=3,
                    label="truth mean")
        ax.plot(k, curves[key].mean(axis=0), color=color(key), lw=2.4, zorder=4,
                label=f"mean of {n_tot} snapshots")
        ax.set(xscale=xscale, yscale="log", ylim=ylims,
               xlabel="indice de mode  k", title=F.SOURCES[key]["label"])
    axes[0].set_ylabel(f"P(k)   [{unit}]")
    axes[-1].plot([], [], color=color(keys[-1]), lw=1.0, alpha=0.35,
                  label=f"{len(idx)} snapshots drawn at random")
    axes[-1].legend(loc="lower left", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    out = save_figure(out, fig)
    plt.close(fig)
    return out


def fig_G(spectra, keys, k, xscale, ylims, idx, out):
    """PSD of G = ∫ I dφ: one panel per source, the angular integral only."""
    return _panels(spectra, keys, k, xscale, ylims, UNIT_G, idx, out)


def fig_direction(spectra, keys, k, m, phi, xscale, ylims, idx, out):
    """PSD of I in ONE direction, the same four panels. spectra[key] is
    (n_snapshots, n_dirs, nbins), directions sorted by phi."""
    return _panels({c: spectra[c][:, m] for c in spectra}, keys, k, xscale, ylims,
                   UNIT_I, idx, out, suptitle=f"φ = {phi:.1f}°")
