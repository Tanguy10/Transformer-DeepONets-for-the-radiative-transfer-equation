import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_SRC))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SRC]
sys.path.insert(0, _PARENT)
_spec = importlib.util.spec_from_file_location("rr_utils", os.path.join(_SRC, "utils.py"))
U = importlib.util.module_from_spec(_spec)
sys.modules["rr_utils"] = U
_spec.loader.exec_module(U)

GRID, N_DIRS = U.GRID, U.N_DIRS
BASE = {"tbd": "Transformer-Based DeepONet", "pca": "PCA-DeepONet", "ae": "AE-DeepONet"}


def _label(fam, res):
    """Transformer-Based DeepONet: native. PCA/AE: working resolution (fixed dimension)."""
    return BASE[fam] if fam == "tbd" else f"{BASE[fam]} @{res}^2"


def parse_model_specs(specs, defaults):
    """'fam:name[:res]' -> list of (key, family, name, res). res = working resolution of
    the fixed-dimension models (150 = upsampled inputs, 50 = everything at 50^2)."""
    out = []
    for s in (specs or defaults):
        parts = s.split(":")
        fam, name = parts[0], parts[1]
        res = int(parts[2]) if len(parts) > 2 else (GRID if fam == "tbd" else GRID)
        out.append((f"{fam}{res}" if fam != "tbd" else fam, fam, name, res))
    return out


def load_models(specs, device):
    models = {}
    for key, fam, name, res in specs:
        try:
            if fam == "tbd":
                op, enc = U.load_tbd(name, device)
                models[key] = (op, enc, "tbd", res, fam)
            else:
                models[key] = (U.load_modular(name, device),
                               U.load_encoder_for(fam, device, res=res, name=name),
                               "modular", res, fam)
            print(f"[loaded] {_label(fam, res):26s} <- {name}")
        except FileNotFoundError as e:
            print(f"[skip] {fam}@{res} ({name}) : {e}")
    return models


@torch.no_grad()
def predict(model, enc, arch, res, T, k, y150, device, src_res=GRID, chunk=40000):
    """I at the 150^2 coords from (T, kappa) supplied at `src_res`.

    PCA/AE first resize the input to their working resolution `res`."""
    if arch == "tbd":
        bi = [U.tbd_encode_native(enc, T, k, src_res, device).to(device), None]
    else:
        Tin = U.resize_fields(T, src_res, res); kin = U.resize_fields(k, src_res, res)
        e = enc.transform_dicts({"u1_in": Tin, "u2_in": kin}, ["u1_in", "u2_in"])
        bi = [e["u1_in"].to(device), e["u2_in"].to(device)]
    return torch.cat([model(bi, y150[c:c + chunk]) for c in range(0, y150.shape[0], chunk)], 1)


@torch.no_grad()
def evaluate_mixed(models, art, clean_I, y150, device, batch=16, limit=0, which="test"):
    """Errors with each input supplied at its degraded resolution (target: 150^2 I),
    per input resolution and overall."""
    groups = art["groups"]
    keep = U.split_mask(art["meta"]["N"], which, art["meta"]["split_seed"])
    nc = GRID * GRID
    res_list = sorted(groups)
    out = {k: {"per_res": {}, "_I": [], "_G": []} for k in models}
    counts = {}
    for R in res_list:
        gr = groups[R]
        val_local = keep[gr["idx"]].nonzero(as_tuple=False).squeeze(1)
        if limit:
            val_local = val_local[:limit]
        counts[R] = int(val_local.numel())
        for key, (model, enc, arch, res_model, fam) in models.items():
            model.eval()
            rI, rG = [], []
            for b0 in range(0, val_local.numel(), batch):
                bl = val_local[b0:b0 + batch]
                pred = predict(model, enc, arch, res_model, gr["T"][bl], gr["kappa"][bl],
                               y150, device, src_res=R)
                true = clean_I[gr["idx"][bl]].to(device)
                rI.append(U.rel_l2_rows(pred, true).cpu())
                rG.append(U.rel_l2_rows(U.to_G(pred, nc), U.to_G(true, nc)).cpu())
            iC, gC = torch.cat(rI), torch.cat(rG)
            out[key]["per_res"][R] = {"I_l2": float(iC.mean()), "G_l2": float(gC.mean()),
                                      "I_l2_std": float(iC.std()), "I_l2_p90": float(
                                          iC.quantile(0.9)) if iC.numel() else float("nan")}
            out[key]["_I"].append(iC); out[key]["_G"].append(gC)
            print(f"  [input {R:3d}^2] {_label(models[key][4], models[key][3]):26s} "
                  f"I-L2 = {float(iC.mean())*100:6.2f}%")
    for key, (model, enc, arch, res_model, fam) in models.items():
        out[key].update({"label": _label(fam, res_model), "family": fam,
                         "input_res": res_model,
                         "I_l2": float(torch.cat(out[key]["_I"]).mean()),
                         "G_l2": float(torch.cat(out[key]["_G"]).mean()),
                         "n_val": sum(counts.values())})
        del out[key]["_I"], out[key]["_G"]
    return out, res_list, counts


@torch.no_grad()
def evaluate_res_sweep(models, T_val, k_val, I_val, y150, device, res_list, batch=16):
    """Errors with the whole split degraded to each resolution of `res_list`."""
    nc = GRID * GRID
    out = {k: {"per_res": {}, "_I": [], "_G": []} for k in models}
    counts = {}
    for R in res_list:
        counts[R] = int(T_val.shape[0])
        T_R = U.resize_fields(T_val, GRID, R)
        k_R = U.resize_fields(k_val, GRID, R)
        for key, (model, enc, arch, res_model, fam) in models.items():
            model.eval()
            rI, rG = [], []
            for b0 in range(0, T_R.shape[0], batch):
                sl = slice(b0, b0 + batch)
                pred = predict(model, enc, arch, res_model, T_R[sl], k_R[sl], y150,
                               device, src_res=R)
                true = I_val[sl].to(device)
                rI.append(U.rel_l2_rows(pred, true).cpu())
                rG.append(U.rel_l2_rows(U.to_G(pred, nc), U.to_G(true, nc)).cpu())
            iC, gC = torch.cat(rI), torch.cat(rG)
            out[key]["per_res"][R] = {"I_l2": float(iC.mean()), "G_l2": float(gC.mean()),
                                      "I_l2_std": float(iC.std()),
                                      "I_l2_p90": float(iC.quantile(0.9))}
            out[key]["_I"].append(iC); out[key]["_G"].append(gC)
            print(f"  [input {R:3d}^2] {_label(fam, res_model):26s} "
                  f"I-L2 = {float(iC.mean())*100:6.2f}%   G-L2 = {float(gC.mean())*100:6.2f}%")
    for key, (model, enc, arch, res_model, fam) in models.items():
        out[key].update({"label": _label(fam, res_model), "family": fam,
                         "input_res": res_model,
                         "I_l2": float(torch.cat(out[key]["_I"]).mean()),
                         "G_l2": float(torch.cat(out[key]["_G"]).mean()),
                         "n_val": counts[res_list[0]]})
        del out[key]["_I"], out[key]["_G"]
    return out, res_list, counts


def format_table_mixed(res, res_list, counts, title=None):
    hdr = (f"{'model':26s} | " + " | ".join(f"{('input ' + str(R)):>11s}" for R in res_list)
           + f" | {'global':>8s}")
    lines = ["", title or "=== MULTI-RESOLUTION INPUT TEST -- target = true 150^2 solver I ===",
             "  (columns = the resolution at which (T, kappa) is given to the model at TEST)",
             "  Transformer-Based DeepONet: native ingestion. PCA/AE @R^2: the input is "
             "brought to R^2 (fixed dim).", ""]
    for metric, lab in (("I_l2", "I-L2"), ("G_l2", "G-L2")):
        lines += [f"  [{lab}]", hdr, "-" * len(hdr)]
        for k in sorted(res, key=lambda f: res[f][metric]):
            row = f"{res[k]['label']:26s} | " + " | ".join(
                f"{res[k]['per_res'][R][metric]*100:10.2f}%" for R in res_list)
            lines.append(row + f" | {res[k][metric]*100:7.2f}%")
        lines.append("")
    lines.append("n snapshots val : " + ", ".join(f"{R}²={counts[R]}" for R in res_list))
    return "\n".join(lines)


def _pick_snapshots(dd, res_list, k_per_res, snaps, which="test"):
    """Snapshots drawn at every resolution -> (global indices, [(R, global index)])."""
    if snaps:
        gidx = [int(s) for s in snaps]
    else:
        from utils import resolve_snapshots
        val_idx = U.split_indices(dd["u1_in"].shape[0])[which]
        sub = {k: dd[k][val_idx] for k in ("u1_in", "u2_in", "I_target")}
        sub["y"] = dd["y"]          # shared grid, never indexed by snapshot
        gidx = [int(val_idx[i]) for i in resolve_snapshots(f"structure:{k_per_res}", sub)]
    return gidx, [(R, g) for g in gidx for R in res_list]


@torch.no_grad()
def fields_figure(dd, idx, res_list=(25, 75, 150), tag=""):
    """Input fields of one snapshot, T and kappa, at each resolution of the campaign."""
    fig, axes = plt.subplots(2, len(res_list), figsize=(3.0 * len(res_list), 6.4),
                             squeeze=False)
    kw = dict(origin="lower", interpolation="nearest", cmap="magma")
    rows = ((dd["u1_in"][idx:idx + 1].float(), "T  [K]"),
            (dd["u2_in"][idx:idx + 1].float(), r"$\kappa$  [m$^{-1}$]"))
    ims = []
    for i, (f150, _) in enumerate(rows):
        vmin, vmax = float(f150.min()), float(f150.max())
        for j, R in enumerate(res_list):
            f = U.avgpool_fields(f150, U.RES_FACTORS[R]).numpy().reshape(R, R)
            im = axes[i][j].imshow(f, vmin=vmin, vmax=vmax, **kw)
            if i == 0:
                axes[0][j].set_title(f"{R}^2", fontsize=9)
        ims.append(im)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for i, (_, label) in enumerate(rows):
        fig.colorbar(ims[i], ax=list(axes[i]), fraction=0.02, pad=0.02, label=label)
    p = os.path.join(U.RESULTS_DIR, f"T_kappa_resolutions_id{idx}{tag}.pdf")
    p = U.save_figure(p, fig, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {os.path.basename(p)}")
    return p


@torch.no_grad()
def snapshot_maps(models, dd, y150, device, res_list, k_per_res=1, snaps=None, tag="",
                  which="test"):
    """One figure per (snapshot, input resolution): each model's predicted G
    and |dG|/||G||_inf, on fixed scales."""
    nc = GRID * GRID
    keys = list(models)
    paths = []
    gsel, jobs = _pick_snapshots(dd, res_list, k_per_res, snaps, which)
    G_ref = {g: U.to_G(dd["I_target"][g:g + 1].float(), nc)[0].numpy().reshape(GRID, GRID)
             for g in gsel}
    vG = float(max(G.max() for G in G_ref.values()))
    vE = float(U.ERR_VMAX or U.ERR_VMAX_DEFAULT)
    print(f"  shared colour scales: G in [0, {vG:.3e}], error in [0, {vE:.0f} %]")

    for R, gidx in jobs:
        Tn = U.resize_fields(dd["u1_in"][gidx:gidx + 1], GRID, R)
        kn = U.resize_fields(dd["u2_in"][gidx:gidx + 1], GRID, R)
        G_true = G_ref[gidx]
        preds = {}
        for key, (model, enc, arch, res_model, fam) in models.items():
            model.eval()
            p = predict(model, enc, arch, res_model, Tn, kn, y150, device, src_res=R)
            preds[key] = U.to_G(p, nc)[0].cpu().numpy().reshape(GRID, GRID)

        fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 6.4), squeeze=False)
        kw = dict(origin="lower", interpolation="nearest")
        g_inf = max(float(np.abs(G_true).max()), 1e-12)
        errs = {k: np.abs(preds[k] - G_true) / g_inf * 100 for k in keys}

        for j, key in enumerate(keys):
            imG = axes[0][j].imshow(preds[key], cmap="inferno", vmin=0, vmax=vG, **kw)
            axes[0][j].set_title(res_label(models, key), fontsize=9)
            imE = axes[1][j].imshow(errs[key], cmap=U.ABS_ERR_CMAP, vmin=0, vmax=vE, **kw)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(imG, ax=list(axes[0]), fraction=0.02, pad=0.02, label="G",
                     extend="max" if max(p.max() for p in preds.values()) > vG else "neither")
        fig.colorbar(imE, ax=list(axes[1]), fraction=0.02, pad=0.02,
                     label="|dG| / ||G||_inf  [%]",
                     extend="max" if max(e.max() for e in errs.values()) > vE else "neither")
        p = os.path.join(U.RESULTS_DIR, f"snapshot_res{R}_id{gidx}{tag}.pdf")
        p = U.save_figure(p, fig, bbox_inches="tight"); plt.close(fig)
        print(f"  -> {os.path.basename(p)}")
        paths.append(p)
    return paths


def res_label(models, key):
    """Label of model `key` (family + working resolution)."""
    return _label(models[key][4], models[key][3])


@torch.no_grad()
def evaluate(models, T_val, k_val, I_val, y150, device, batch=16):
    """Relative L2 errors (mean of ratios) on I and G, full-resolution split."""
    n = T_val.shape[0]
    nc = GRID * GRID
    out = {}
    for key, (model, enc, arch, res, fam) in models.items():
        model.eval()
        rI, rG = [], []
        for b0 in range(0, n, batch):
            sl = slice(b0, b0 + batch)
            pred = predict(model, enc, arch, res, T_val[sl], k_val[sl], y150, device)
            true = I_val[sl].to(device)
            rI.append(U.rel_l2_rows(pred, true).cpu())
            rG.append(U.rel_l2_rows(U.to_G(pred, nc), U.to_G(true, nc)).cpu())
        iC, gC = torch.cat(rI), torch.cat(rG)
        out[key] = {"label": _label(fam, res), "family": fam, "input_res": res,
                    "I_l2": float(iC.mean()), "G_l2": float(gC.mean()),
                    "I_l2_std": float(iC.std()), "G_l2_std": float(gC.std()), "n_val": n}
        print(f"  {out[key]['label']:26s} I-L2 = {out[key]['I_l2']*100:6.2f}%   "
              f"G-L2 = {out[key]['G_l2']*100:6.2f}%")
    return out


def format_table(res, n_val):
    hdr = f"{'model':26s} | {'I-L2':>9s} | {'G-L2':>9s}"
    lines = ["", "=== Trained on imperfect data (150/75/25) -- "
                 "TESTED on CLEAN 150x150 data (input AND label) ===",
             f"  mean relative L2 error over the {n_val} snapshots of the split",
             "  (@R^2 = working resolution imposed by the model's fixed dimension)", hdr,
             "-" * len(hdr)]
    for k in sorted(res, key=lambda f: res[f]["I_l2"]):
        lines.append(f"{res[k]['label']:26s} | {res[k]['I_l2']*100:8.2f}% | {res[k]['G_l2']*100:8.2f}%")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Test the models on a CLEAN 150x150 split (input AND label) -- "
                    "TEST by default")
    ap.add_argument("--model", action="append", metavar="FAM:NAME[:RES]",
                    help="model to evaluate, repeatable. RES = working resolution of the "
                         "fixed-dimension models (default 150). E.g. --model tbd:res-TBD "
                         "--model pca:robPCA50:50")
    ap.add_argument("--clean", default=U.CLEAN_DATASET)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Split to evaluate. TEST by default: the val split was used to "
                         "pick the best checkpoint during training.")
    ap.add_argument("--test", default="clean", choices=["clean", "mixed"],
                    help="'clean': CLEAN 150^2 test inputs (default). "
                         "'mixed': MULTI-RESOLUTION test inputs (150^2/75^2/25^2), "
                         "target = true 150^2 I in both cases.")
    ap.add_argument("--input_res", type=int, action="append", metavar="R",
                    help="UNSEEN-RESOLUTION test, repeatable: the clean split is degraded "
                         "from 150^2 to R^2 and given to every model, the target staying "
                         "the true 150^2 I. Every column is measured on ALL the snapshots "
                         "(unlike --test mixed, whose columns are disjoint thirds). R must "
                         "be a multiple of the encoder patch (5) for the Transformer-Based "
                         "DeepONet. E.g. --input_res 100")
    ap.add_argument("--tag", default="", help="suffix of the figure files (overwrites nothing)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--snapshots", type=int, default=0, metavar="K",
                    help="draw K snapshots, EACH at every input resolution (150^2/75^2/25^2): "
                         "one column per model, its prediction G and its |dG| map")
    ap.add_argument("--snap", type=int, action="append", metavar="IDX",
                    help="explicit snapshot index/indices to draw (repeatable); each is "
                         "supplied at ITS resolution in the mixed dataset")
    ap.add_argument("--fields", type=int, action="append", metavar="IDX",
                    help="draw the INPUT fields of snapshot IDX -- T and kappa, one row "
                         "each -- at every resolution of the campaign (150^2/75^2/25^2), "
                         "then stop: no model is loaded")
    ap.add_argument("--figs_only", action="store_true",
                    help="only draw the snapshots, no table computed")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(U.RESULTS_DIR, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""

    dd = torch.load(args.clean, map_location="cpu", mmap=True, weights_only=False)["dict_data"]
    if args.fields:
        print("\n[input fields per resolution -- T and kappa]")
        for i in args.fields:
            fields_figure(dd, i, tag=sfx)
        print(f"\n[OK] Figures -> {U.RESULTS_DIR}")
        return
    N = dd["u1_in"].shape[0]
    val_idx = U.split_indices(N)[args.split]
    if args.limit:
        val_idx = val_idx[:args.limit]
    T_val, k_val, I_val = dd["u1_in"][val_idx], dd["u2_in"][val_idx], dd["I_target"][val_idx]
    y150 = dd["y"].float().to(device)
    print(f"Split {args.split.upper()} PROPRE : {val_idx.numel()} snapshots, "
          f"input and label at {GRID}x{GRID}")

    specs = parse_model_specs(args.model, ["tbd:res-TBD", "pca:robPCA", "ae:robAE"])
    models = load_models(specs, device)
    if not models:
        print("No model loaded."); return

    art = None
    if args.test == "mixed":
        U.build_mixed()
        art = U.load_mixed()

    if args.figs_only:
        print("\n[snapshots, each at every input resolution -- figures only]")
        snapshot_maps(models, dd, y150, device, sorted(U.RES_FACTORS),
                      k_per_res=max(1, args.snapshots), snaps=args.snap, tag=sfx,
                      which=args.split)
        print(f"\n[OK] Figures -> {U.RESULTS_DIR}")
        return

    if args.input_res:
        res_list = sorted(set(int(r) for r in args.input_res))
        print("\n[evaluation -- clean split degraded to each requested resolution, "
              "target = true 150^2 I]")
        res, res_list, counts = evaluate_res_sweep(models, T_val, k_val, I_val, y150,
                                                   device, res_list)
        table = format_table_mixed(
            res, res_list, counts,
            title="=== UNSEEN-RESOLUTION TEST -- target = true 150^2 solver I ===")
    elif args.test == "mixed":
        print("\n[evaluation -- multi-resolution TEST inputs, target = true 150^2 I]")
        res, res_list, counts = evaluate_mixed(models, art, dd["I_target"], y150,
                                               device, limit=args.limit,
                                               which=args.split)
        table = format_table_mixed(res, res_list, counts)
    else:
        print("\n[evaluation -- same clean 150^2 input, seen at each model's working resolution]")
        res = evaluate(models, T_val, k_val, I_val, y150, device)
        table = format_table(res, val_idx.numel())
    print(table)
    if args.snapshots or args.snap:
        print(f"\n[snapshots, each at every input resolution]")
        snapshot_maps(models, dd, y150, device, sorted(U.RES_FACTORS),
                      k_per_res=max(1, args.snapshots), snaps=args.snap, tag=sfx,
                      which=args.split)
        print(f"\n[OK] Figures -> {U.RESULTS_DIR}")


if __name__ == "__main__":
    main()
