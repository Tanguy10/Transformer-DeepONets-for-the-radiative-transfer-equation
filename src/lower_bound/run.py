import argparse
import contextlib
import io
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (ImportError, AttributeError, ValueError):
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_RTE2D = os.path.dirname(_HERE)
for _p in (_RTE2D, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bound
import model_errors
from utils import DEFAULT_DATASET, save_figure

CACHE = os.path.join(_HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

"""`p` = dimension of the reconstructor: p_basis for the DeepONets (the branch.trunk
output), n_latent * dim for the Transformer-Based DeepONet, whose attention decoder
reads the whole latent. The Thm 3.4 bound only CONSTRAINS the affine rows; for the
Transformer-Based DeepONet, p merely places it at a comparable width.

Both are read off the checkpoints by `model_errors.discover_models`: the models compared
are the ones sitting in --ckpt_dir (checkpoints/ by default), never a list frozen in this
file. The AFFINE runs (`<run>-affine`, no output softplus -- the only ones Thm 3.4 covers
to the letter) sit there too, and several directories may be given to bring in the
ablations of a sub-experiment."""
FN = {"pca": model_errors.pca_deeponet, "autoencoder": model_errors.ae_deeponet,
      "tbd": model_errors.tbd_operator}
SETS = {"train": "TRAIN", "test": "TEST"}

FAMILY = {"pca": ("#2f6fb2", "o", 8.0, "PCA-DeepONet"),
          "autoencoder": ("#d0761c", "s", 7.2, "AE-DeepONet"),
          "tbd": ("#2e8b57", "^", 9.0, "Transformer-Based DeepONet")}
KINDS = ("pca", "autoencoder", "tbd")
FS = 1.3
INK, INK2 = "#111111", "#5b5a57"
BAND = "#f1efe9"


@contextlib.contextmanager
def quiet(enabled=True):
    """Silence everything the dependencies print."""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _fresh(path, M, refresh, tag, verbose):
    """The cached array, unless missing, stale (length != M) or forced."""
    if refresh or not os.path.exists(path):
        return None
    a = np.load(path)
    if len(a) == M:
        return a
    if verbose:
        print(f"[{tag}] stale cache ({len(a)} != {M} snapshots) -> recomputing", flush=True)
    return None


def get_spectrum(I_S, tag, refresh, verbose=False):
    path = os.path.join(CACHE, f"lam_{tag}.npy")
    lam = _fresh(path, I_S.shape[0], refresh, tag, verbose)
    if lam is not None:
        return {"lam": lam, "M": len(lam), "d": None}
    if verbose:
        print(f"[{tag}] SVD of the centred data matrix ...", flush=True)
    spec = bound.eigenvalues(I_S, tag=tag, verbose=verbose)
    np.save(path, spec["lam"])
    return spec


def get_errors(splits, device, tag, refresh, models, verbose=False):
    """||I_m - N(u_m)||^2 per snapshot for each model of the `tag` split, cached."""
    M = model_errors.split_data(splits, tag)["I_target"].shape[0]
    out = {}
    for name, kind, _p, _lin, ckpt_dir in models:
        npz = os.path.join(CACHE, f"metrics_{name}_{tag}.npz")
        npy = os.path.join(CACHE, f"sqerr_{name}_{tag}.npy")
        se = None
        if not refresh and os.path.exists(npz):
            d = np.load(npz)["sqerr"]
            if len(d) == M:
                se = d
            elif verbose:
                print(f"[{tag}/{name}] stale cache ({len(d)} != {M}) -> recomputing",
                      flush=True)
        if se is None:
            se = _fresh(npy, M, refresh, f"{tag}/{name}", verbose)
        if se is None:
            if verbose:
                print(f"[{tag}] E_hat: {name} ...", flush=True)
            se = FN[kind](splits, device, tag, name, ckpt_dir=ckpt_dir)
            np.save(npy, se)
        out[name] = se
    return out


def p_star(curve, value):
    """p* = the first p at which the bound drops below `value`."""
    pp, bb = curve
    under = bb <= value
    return int(pp[np.argmax(under)]) if under.any() else None


def reference(rows, t):
    """The model the horizontal is drawn at: the narrowest nonlinear one."""
    cands = [r for r in rows if not r["affine"] and r["ehat"][t] is not None]
    return min(cands, key=lambda r: (r["p"], r["ehat"][t]), default=None)


def _label(name):
    """Marker tag: the part of the name that tells two models of a family apart."""
    from encoders import encoder_run
    head = encoder_run(name).split("-", 1)
    return head[1] if len(head) == 2 and head[0].upper() in ("PCA", "AE", "TBD") else name


def _place(fig, ax, items, blockers, fontsize=8.2 * FS):
    """Place the labels where they overlap nothing already drawn."""
    cands = [(dx * FS, dy * FS) for dx, dy in
             [(9, 4), (9, -6), (-9, 4), (-9, -6), (0, 10), (0, -14), (14, 11), (-14, -15),
              (0, -32), (18, -30), (-18, -30), (0, 26), (22, 14), (-22, 14)]]
    rend = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer=rend)
    taken = list(blockers)

    def cost(bb):
        out = (max(0, frame.x0 - bb.x0) + max(0, bb.x1 - frame.x1)
               + max(0, frame.y0 - bb.y0) + max(0, bb.y1 - frame.y1))
        hit = sum((min(bb.x1, o.x1) - max(bb.x0, o.x0)) * (min(bb.y1, o.y1) - max(bb.y0, o.y0))
                  for o in taken if bb.overlaps(o))
        return 1e3 * out + hit

    def draw(x, y, text, color, off):
        dx, dy = off
        return ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                           fontsize=fontsize, color=color, zorder=8,
                           ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                           va="bottom" if dy >= 0 else "top")

    for x, y, text, color in items:
        best = (None, None, None)
        for off in cands:
            a = draw(x, y, text, color, off)
            bb = a.get_window_extent(renderer=rend).expanded(1.08, 1.35)
            c = cost(bb)
            if c == 0:
                best = (0, off, bb)
                break
            a.remove()
            if best[0] is None or c < best[0]:
                best = (c, off, bb)
        else:
            draw(x, y, text, color, best[1])
        taken.append(best[2])
    return taken


def _marker_boxes(ax, pts, pad=7.0):
    """The display boxes the markers occupy, so the labels can dodge them."""
    from matplotlib.transforms import Bbox
    out = []
    for x, y in pts:
        px, py = ax.transData.transform((x, y))
        out.append(Bbox.from_extents(px - pad, py - pad, px + pad, py + pad))
    return out


def _curve_boxes(ax, xs, ys, n=80, pad=3.0):
    """Display boxes along the visible part of the bound."""
    from matplotlib.transforms import Bbox
    if len(xs) == 0:
        return []
    step = max(1, len(xs) // n)
    out = []
    for px, py in ax.transData.transform(list(zip(xs[::step], ys[::step]))):
        out.append(Bbox.from_extents(px - pad, py - pad, px + pad, py + pad))
    return out


def _bar_box(ax, x0, x1, y, pad=3.5):
    """The display box of the horizontal, so the labels do not lie across it."""
    from matplotlib.transforms import Bbox
    (a, yy), (b, _) = ax.transData.transform([(x0, y), (x1, y)])
    lo, hi = sorted((a, b))
    return Bbox.from_extents(lo - pad, yy - pad, hi + pad, yy + pad)


def plot_bound_comparison(spec, curves, rows, out_png):
    """The bound sum_{k>p} lambda_k against p, each model at its own width p.

    Ordinate: E_hat^2 = (1/M) sum_m ||I_m - G(T_m, kappa_m)||^2, unnormalised."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

    kinds = [k for k in KINDS if any(r["kind"] == k for r in rows)]
    p_min, p_max = min(r["p"] for r in rows), max(r["p"] for r in rows)
    x_lo, x_hi = min(100.0, 0.9 * p_min), 1.45 * p_max
    sq = lambda v: v * v
    ehats = [sq(e) for r in rows for e in r["ehat"].values() if e]
    tops = [sq(curves[t][1][curves[t][0] >= x_lo][0]) for t in SETS]
    y_lo = min(sq(2e4), 0.9 * min(ehats))
    y_hi = 1.3 * max(ehats + tops)

    fig, axes = plt.subplots(1, len(SETS), figsize=(12.6, 5.0), sharey=True,
                             squeeze=False)
    first = next(iter(SETS))
    pending = []
    for ax, t in zip(axes[0], SETS):
        pp, bb = curves[t]
        sl = (pp >= x_lo * 0.9) & (pp <= x_hi)
        ref = reference(rows, t) if t == first else None
        ax.fill_between(pp[sl], y_lo, sq(bb[sl]), color=BAND, zorder=0.5, lw=0)
        ax.plot(pp[sl], sq(bb[sl]), lw=2.2, color=INK2, zorder=3,
                solid_capstyle="round")
        curve = (pp[sl], sq(bb[sl]))

        pts, items = [], []
        for r in rows:
            col, mk, ms, _ = FAMILY[r["kind"]]
            if r["ehat"][t] is None:
                continue
            e = sq(r["ehat"][t])
            x = r["p"]
            ax.plot([x], [e], mk, ms=ms, mfc=col, mec="white", mew=1.1, zorder=6)
            pts.append((x, e))
            lbl = _label(r["name"])
            if lbl.lstrip("pP") != str(r["p"]):
                items.append((x, e, lbl, col))

        bar = None
        if ref is not None:
            col = FAMILY[ref["kind"]][0]
            ps = p_star(curves[t], ref["ehat"][t])
            e = sq(ref["ehat"][t])
            if ps is not None and ps <= x_hi:
                ax.plot([ref["p"], ps], [e, e], lw=1.1, ls=(0, (4, 3)), color=col,
                        alpha=0.65, zorder=4)
                ax.plot([ps], [e], "o", ms=7.5, mfc="white", mec=col, mew=1.8, zorder=7)
                pts.append((ps, e))
                bar = (ref["p"], ps, e)
                items.append((ps, e, f"$p^\\star$ = {ps}", col))

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        widths = sorted({r["p"] for r in rows})
        if len(widths) <= 8:
            ax.set_xticks(widths)
            ax.set_xticklabels([str(w) for w in widths])
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        ax.yaxis.set_major_formatter(
            LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{SETS[t]}   (M = {spec[t]['M']})", fontsize=11.5 * FS, color=INK,
                     pad=10, fontweight="medium")
        ax.set_xlabel("$p$  --  dimension of the reconstructor", fontsize=10 * FS, color=INK2)
        ax.grid(False)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d6d5d0")
        ax.tick_params(colors=INK2, labelsize=9 * FS)
        ax.tick_params(which="major", length=5.0, width=0.9)
        ax.tick_params(which="minor", length=2.8, width=0.7)
        leg = None
        if t == first:
            handles = [Line2D([0], [0], color=INK2, lw=2.2)]
            handles += [Line2D([0], [0], marker=FAMILY[k][1], color="none", ls="none",
                               mfc=FAMILY[k][0], mec="white", mew=1.1, ms=FAMILY[k][2])
                        for k in kinds]
            labels = ["Lower bound"] + [FAMILY[k][3] for k in kinds]
            leg = ax.legend(handles, labels, loc="upper right",
                            frameon=False, fontsize=9 * FS, labelcolor=INK, handlelength=1.8,
                            handletextpad=0.6, borderaxespad=0.6, labelspacing=0.6)
            leg.set_zorder(9)

        items.sort(key=lambda it: -len(it[2]))
        pending.append((ax, items, pts, bar, leg, curve))
    ax0 = axes[0][0]
    ax0.set_ylabel("L2 error$^2$", rotation=0, ha="left", va="bottom",
                   fontsize=10.5 * FS, color=INK2)
    ax0.yaxis.set_label_coords(-0.085, 1.015)

    fig.tight_layout()
    # After tight_layout: label placement works in display coordinates, which it shifts.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, items, pts, bar, leg, curve in pending:
        blockers = _marker_boxes(ax, pts)
        if bar is not None:
            blockers.append(_bar_box(ax, *bar))
        if leg is not None:
            blockers.append(leg.get_window_extent(renderer=rend).expanded(1.04, 1.1))
        blockers += _curve_boxes(ax, *curve)
        _place(fig, ax, items, blockers)
    out_png = save_figure(out_png, fig, facecolor="white")
    plt.close(fig)
    return out_png


def table_bound(rows):
    """E_hat against the bound at each model's width p, on train and test."""
    W = 24
    hdr = (f"{'model':<{W}}{'p':>6}" +
           "".join(f"{'E ' + SETS[t]:>12}{'bound':>11}{'E/bound':>9}{'p*':>6}"
                   for t in SETS))
    sep = "─" * len(hdr)
    L = [hdr, sep]
    for r in rows:
        line = f"{r['name']:<{W}}{r['p']:>6}"
        for t in SETS:
            e, b, ps = r["ehat"][t], r["bound"][t], r["p_star"][t]
            if e is None:
                line += f"{'-':>12}{b:>11.3e}{'-':>9}{'-':>6}"
            elif b <= 0:
                line += f"{e:>12.3e}{b:>11.3e}{'-':>9}{(ps if ps else '-'):>6}"
            else:
                mark = "" if r["affine"] else "†"
                line += (f"{e:>12.3e}{b:>11.3e}{e / b:>8.2f}{mark:<1}"
                         f"{(ps if ps is not None else '>M'):>6}")
        L.append(line)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description="Compare the trained models with the Thm 3.4 lower bound.",
        epilog="e.g.  python run.py --models 'PCA-*' 'TBD-N?-D128' --out tbd_vs_pca.pdf")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt_dir", nargs="+", default=None, metavar="DIR",
                    help="where the models are read from, one or more directories "
                         "(default: checkpoints/). Add dot_decoder/checkpoints to "
                         "bring the junction ablation into the comparison.")
    ap.add_argument("--models", nargs="+", default=None, metavar="PATTERN",
                    help="models to compare, by name or shell pattern "
                         "(default: every model of --ckpt_dir)")
    ap.add_argument("--list", action="store_true",
                    help="list the models found in --ckpt_dir and exit")
    ap.add_argument("--out", default=None, help="output figure (default: %(default)s)")
    ap.add_argument("--refresh", choices=["none", "spectrum", "errors", "all"], default="none")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print diagnostics (progress, numerical checks)")
    ap.set_defaults(out=os.path.join(_HERE, "bound_comparison.pdf"))
    args = ap.parse_args()

    ref_sp = args.refresh in ("spectrum", "all")
    ref_er = args.refresh in ("errors", "all")

    dirs = args.ckpt_dir or [model_errors.default_ckpt_dir()]
    found = model_errors.discover_models(dirs)
    if not found:
        raise SystemExit(f"no model found in {', '.join(dirs)}")
    if args.list:
        print(f"{len(found)} models in {', '.join(dirs)}")
        for name, kind, p, affine, _d in found:
            head = ("cross-attention, no softplus" if kind == "tbd"
                    and model_errors.affine_head(name) else "cross-attention" if kind == "tbd"
                    else "affine (Thm 3.4)" if model_errors.affine_head(name)
                    else "affine + softplus")
            print(f"  {name:<24} {kind:<12} p = {p:<5} {head}")
        return
    models = model_errors.select_models(found, args.models)

    with quiet(not args.verbose):
        import torch
        from utils import load_and_split_dataset
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        splits = load_and_split_dataset(
            os.path.join(_RTE2D, args.dataset) if not os.path.isabs(args.dataset)
            else args.dataset, map_location="cpu")
        I = {t: model_errors.split_data(splits, t)["I_target"] for t in SETS}
        n_split = {k: len(splits[f"{k}_idx"]) for k in ("train", "val", "test")}
        spec = {t: get_spectrum(I[t], t, ref_sp, args.verbose) for t in SETS}
        errs = {t: get_errors(splits, device, t, ref_er, models, args.verbose)
                for t in SETS}
        curves = {t: bound.energy_curve(spec[t]) for t in SETS}

    rows = []
    for name, kind, p, affine, _d in models:
        e = {t: (model_errors.E_hat(errs[t][name]) if name in errs[t] else None) for t in SETS}
        rows.append({
            "name": name, "kind": kind, "p": p, "affine": affine, "ehat": e,
            "bound": {t: bound.lower_bound(spec[t], p) for t in SETS},
            "p_star": {t: (p_star(curves[t], e[t]) if e[t] else None) for t in SETS},
        })

    png = args.out
    with quiet(not args.verbose):
        plot_bound_comparison(spec, curves, rows, png)

    aff = [r for r in rows if r["affine"]]
    ok = all(r["ehat"][t] >= r["bound"][t] for r in aff for t in SETS if r["ehat"][t])
    print(f"E^2 = (1/M) sum_m ||I_m - N(u_m)||^2   >=   sum_(k>p) lambda_k   "
          f"if the reconstructor is AFFINE   (Thm 3.4)\n"
          f"{len(rows)} models from {', '.join(dirs)} -- 70/15/15 split, "
          f"train {n_split['train']} / test {n_split['test']} snapshots\n")
    print(table_bound(rows))
    print(f"\n† nonlinear reconstructor: the bound does not constrain it, E/bound < 1 "
          f"means that no\n  DeepONet of that width p can match this model.  "
          f"p* = the width an affine reconstructor\n  would need to hope for this E.  "
          f"The {len(aff)} affine branch.trunk junctions respect the bound -> "
          f"{'VERIFIED' if ok else 'VIOLATED (bug?)'}.")
    print(f"Figure: {png}")


if __name__ == "__main__":
    main()
