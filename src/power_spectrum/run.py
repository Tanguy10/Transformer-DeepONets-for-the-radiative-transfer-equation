#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SRC)
sys.path.insert(0, _HERE)
sys.path.insert(0, _SRC)

import fields as F
import figures
from psd import SpectrumBinner, selftest, top_ring
from utils import DEFAULT_DATASET

CACHE_DIR = os.path.join(_HERE, "cache")
RESULTS_DIR = os.path.join(_HERE, "results")

TAIL_LO = 40.0


def _cache_path(split, key, n):
    return os.path.join(CACHE_DIR, f"{split}_{key}_n{n}.npz")


def compute_source(key, dict_split, geom, y_full, binner, n, device, batch=64,
                   pts_per_fwd=F.PTS_PER_FWD, verbose=False):
    """(P_G, P_I, err_G, err_I) of one source, per snapshot (and per direction for I).

    The errors are None for the truth."""
    model, encoder = F.load_source(key, device)
    is_truth = model is None

    P_G = np.empty((n, binner.nbins), dtype=np.float32)
    P_I = np.empty((n, geom["n_dirs"], binner.nbins), dtype=np.float32)
    err_G = None if is_truth else np.empty(n, dtype=np.float32)
    err_I = None if is_truth else np.empty(n, dtype=np.float32)
    t0 = time.time()

    for i0 in range(0, n, batch):
        sl = slice(i0, min(i0 + batch, n))
        true_flat = F.truth_I(dict_split, sl)
        flat = true_flat if is_truth else F.predict_I(
            model, encoder, dict_split, sl, y_full, device, pts_per_fwd=pts_per_fwd)
        img = F.to_images(flat, geom)
        g = F.to_G(img, geom)
        if not is_truth:
            gt = F.to_G(F.to_images(true_flat, geom), geom)
            err_I[sl] = F.rel_l2(flat, true_flat)
            err_G[sl] = F.rel_l2(g.reshape(g.shape[0], -1),
                                 gt.reshape(gt.shape[0], -1))
        P_G[sl] = binner(g).astype(np.float32)
        P_I[sl] = binner(img).astype(np.float32)
        if verbose:
            done, el = sl.stop, time.time() - t0
            print(f"\r    {key:<6s} {done}/{n}  ({el:.0f} s, "
                  f"eta {el * (n - done) / max(done, 1):.0f} s)", end="", flush=True)
    if verbose:
        print()

    del model, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return P_G, P_I, err_G, err_I


def load_cached(split, key, n, n_dirs, nbins):
    """The cached spectra, or None unless they match this run exactly."""
    p = _cache_path(split, key, n)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    if not {"P_G", "P_I"} <= set(d.files):
        return None
    if d["P_G"].shape != (n, nbins) or d["P_I"].shape != (n, n_dirs, nbins):
        return None
    return d


def select_dirs(phi, wanted):
    """Indices of the directions to plot: all of them, or the nearest available
    ordinate to each requested angle in degrees."""
    if not wanted:
        return list(range(len(phi)))
    out = []
    for w in wanted:
        m = int(np.argmin(np.abs((np.asarray(phi) - w + 180.0) % 360.0 - 180.0)))
        if m not in out:
            out.append(m)
    return out


def tail_ratio(P, P_truth, k, band):
    """Energy of the model relative to the truth over a band of k, per snapshot."""
    m = (k >= band[0]) & (k <= band[1])
    return P[:, m].sum(axis=1) / P_truth[:, m].sum(axis=1)


def main():
    ap = argparse.ArgumentParser(
        description="Power spectral density of the FVDOM truth and of trained models.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--sources", nargs="+", default=["truth", *F.MODEL_KEYS],
                    help="one panel per source, in this order; 'truth' is the "
                         "dashed reference of every panel. A source is a key of "
                         f"the table ({' | '.join(F.SOURCES)}) or any train.py run "
                         "name, prefixed by its family when one name carries "
                         "several (tbd:my-run)")
    ap.add_argument("--figures", default="all", choices=("all", "G", "angles"),
                    help="which family to draw")
    ap.add_argument("--angles", type=float, nargs="+", metavar="DEG",
                    help="degrees to draw instead of all the ordinates; each is "
                         "matched to the nearest one")
    ap.add_argument("--n-snapshots", type=int, default=0, help="0 = whole block")
    ap.add_argument("--refresh", default="none",
                    help="'none', 'all', or a comma-separated list of sources")
    ap.add_argument("--kmax", default="nyquist",
                    choices=("complete", "nyquist", "corner"),
                    help="last plotted ring: 'complete' = npix/2-1 (last full "
                         "ring), 'nyquist' = npix/2, 'corner' = every sampled "
                         "mode, down to a few percent of angular coverage")
    ap.add_argument("--xscale", default="linear", choices=("log", "linear"),
                    help="scale of the k axis only; P(k) stays logarithmic")
    ap.add_argument("--n-show", type=int, default=120,
                    help="snapshots drawn per panel (the mean always uses them all)")
    ap.add_argument("--tag", default="",
                    help="suffix added to every figure, so a run on another set "
                         "of sources does not overwrite the previous one")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pts-per-fwd", type=int, default=F.PTS_PER_FWD,
                    help="points per forward; lower it if the GPU is busy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    known = set(F.SOURCES)
    args.sources = [F.resolve_source(s) for s in args.sources]
    extra = [k for k in args.sources if k not in known]
    if extra and not args.tag:
        args.tag = extra[0]

    t_start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not args.no_selftest:
        print("Method self-test")
        selftest(verbose=args.verbose)
        if not args.verbose:
            print("  5 checks OK (reference implementation, Parseval, pure mode, "
                  "corner mode, Parseval over the whole square)")

    from utils import load_and_split_dataset
    print(f"\nChargement de {os.path.basename(args.dataset)} (mmap)")
    sp = load_and_split_dataset(args.dataset)
    dict_split = {"train": sp["data"][0], "val": sp["data"][1],
                  "test": sp["data_test"]}[args.split]
    y_full = sp["y_full"]
    geom = F.grid_geometry(y_full)
    phi, order = F.angles_deg(y_full, geom["n_dirs"])
    n_all = int(dict_split["u1_in"].shape[0])
    n = n_all if args.n_snapshots <= 0 else min(args.n_snapshots, n_all)
    device = torch.device(args.device)
    print(f"  bloc {args.split} : {n}/{n_all} snapshots, {geom['n_dirs']} "
          f"directions, grille {geom['grid']}², device {device}")

    # Always binned out to the corners, then sliced: one cache serves any --kmax.
    binner = SpectrumBinner(geom["grid"], kmax="corner")
    keep = binner.kvals <= top_ring(geom["grid"], args.kmax)
    k = binner.kvals[keep]
    tail = (TAIL_LO, float(binner.k_complete))
    print(f"  k = {int(k[0])} … {int(k[-1])} (--kmax {args.kmax})")

    refresh = ([] if args.refresh == "none" else list(F.SOURCES)
               if args.refresh == "all" else args.refresh.split(","))

    print()
    P_G, P_I, err_G, err_I = {}, {}, {}, {}
    for key in args.sources:
        d = (None if key in refresh else
             load_cached(args.split, key, n, geom["n_dirs"], binner.nbins))
        if d is None:
            print(f"  {key:<6s} : calcul ({F.SOURCES[key]['label']})")
            pg, pi, eg, ei = compute_source(
                key, dict_split, geom, y_full, binner, n, device,
                batch=args.batch, pts_per_fwd=args.pts_per_fwd,
                verbose=args.verbose)
            np.savez(_cache_path(args.split, key, n), P_G=pg, P_I=pi,
                     kvals=binner.kvals,
                     **({} if eg is None else {"err_G": eg, "err_I": ei}))
        else:
            print(f"  {key:<6s} : cache")
            pg, pi = d["P_G"], d["P_I"]
            eg = d["err_G"] if "err_G" in d.files else None
            ei = d["err_I"] if "err_I" in d.files else None
        P_G[key] = np.asarray(pg, dtype=np.float64)[:, keep]
        # Sort by phi: the dataset stores the ordinates in arbitrary order.
        P_I[key] = np.asarray(pi, dtype=np.float64)[:, order][..., keep]
        err_G[key], err_I[key] = eg, ei

    if "truth" not in P_G:
        raise SystemExit("The truth is required as a reference: "
                         "add 'truth' to --sources.")

    keys = list(args.sources)
    idx = figures.sample_idx(n, args.n_show)
    tag = f"_{args.tag}" if args.tag else ""
    print("\nFigures →", os.path.relpath(RESULTS_DIR, _REPO))

    if args.figures in ("all", "G"):
        out = os.path.join(RESULTS_DIR, f"psd_G{tag}.pdf")
        out = figures.fig_G(P_G, keys, k, args.xscale, figures.ylim(P_G.values()),
                            idx, out)
        print(f"  {os.path.basename(out)}")

    if args.figures in ("all", "angles"):
        ylims = figures.ylim(P_I.values())
        names = []
        for m in select_dirs(phi, args.angles):
            out = os.path.join(RESULTS_DIR, f"psd_I_phi{int(round(phi[m], 3)):03d}{tag}.pdf")
            out = figures.fig_direction(P_I, keys, k, m, phi[m], args.xscale, ylims,
                                        idx, out)
            names.append(os.path.basename(out))
        print(f"  {names[0]} … {names[-1]}  ({len(names)} figures, "
              "one per direction, shared scales)")

    print()
    for key in keys:
        if err_G[key] is None:
            continue
        r = tail_ratio(P_G[key], P_G["truth"], k, tail)
        rd = np.array([np.median(tail_ratio(P_I[key][:, m], P_I["truth"][:, m],
                                            k, tail))
                       for m in range(P_I[key].shape[1])])
        print(f"  {F.SOURCES[key]['label']:<39s} G-L2 {100 * err_G[key].mean():5.2f} %"
              f"   I-L2 {100 * err_I[key].mean():5.2f} %"
              f"   tail energy (k in [{tail[0]:.0f},{tail[1]:.0f}]) "
              f"G {100 * np.median(r):5.1f} %   I per direction "
              f"{100 * rd.min():.0f}–{100 * rd.max():.0f} % "
              f"(pire : φ = {phi[int(np.argmin(rd))]:.1f}°)")
    print(f"\nDone in {time.time() - t_start:.0f} s.")


if __name__ == "__main__":
    main()
