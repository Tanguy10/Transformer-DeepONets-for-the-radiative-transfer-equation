import contextlib
import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

_SRC = os.path.dirname(os.path.abspath(__file__))
_SUB = os.path.dirname(_SRC)
_PARENT = os.path.dirname(_SUB)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from utils import (ABS_ERR_CMAP, CHECKPOINTS_DIR, DEFAULT_DATASET, ERR_VMAX,
                   ERR_VMAX_DEFAULT, TrainingHistory, plot_learning_curves,
                   rel_phys_loss, save_atomic, save_figure)

CKPT_DIR = os.path.join(_SUB, "checkpoints")
RESULTS_DIR = os.path.join(_SUB, "results")
CLEAN_DATASET = DEFAULT_DATASET
# The derived datasets (tens of GB) sit next to the clean one, wherever that actually
# resolves to -- data/*.pt is a symlink to bulk storage here. Override with
# RTE2D_DERIVED_DATA_DIR.
DERIVED_DIR = os.environ.get("RTE2D_DERIVED_DATA_DIR",
                             os.path.dirname(os.path.realpath(DEFAULT_DATASET)))
MIXED_DATASET = os.path.join(DERIVED_DIR, "mixed_res_dataset_phi.pt")
INTERP150_DATASET = os.path.join(DERIVED_DIR, "interp150_dataset_phi.pt")

# EVERY encoder of this sub-package is fitted on the MIXED-RESOLUTION data of the TRAIN
# split, never on the clean 150^2 dataset of the main campaign: the paper states that the
# three models are trained on exactly the same data, and the encoder is stage one of that
# training. `fit_tbd_encoder_on_mixed` builds the Transformer one, `fit_encoder_at_res`
# the PCA/AE ones (at their own working resolution, 150^2 or 50^2).
ENC_TBD = os.path.join(CKPT_DIR, "encoder_tbd_mixed.pt")

GRID = 150
N_DIRS = 16
SPLIT_SEED = 42
RES_FACTORS = {150: 1, 75: 2, 25: 6}


def avgpool_fields(x_flat, factor, H=GRID):
    """Scalar field (n, H*H) -> (n, (H/f)^2) by MEAN over f x f blocks (avg-pool).
    coarse[i,j] = (1/f^2) sum_{a,b<f} fine[f*i+a, f*j+b]."""
    n = x_flat.shape[0]
    x = x_flat.float().view(n, 1, H, H)
    if factor > 1:
        x = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    return x.reshape(n, -1)


def avgpool_intensity(I_flat, factor, H=GRID):
    """Intensity (n, H*H*16) -> (n, (H/f)^2*16), SPATIAL avg-pool (16 directions kept).
    In: index (c*16+d), c=i*H+j.  Out: (c'*16+d), c'=i'*(H/f)+j'."""
    n = I_flat.shape[0]
    I = I_flat.float().view(n, H, H, N_DIRS).permute(0, 3, 1, 2)
    if factor > 1:
        I = F.avg_pool2d(I, kernel_size=factor, stride=factor)
    Hc = I.shape[-1]
    return I.permute(0, 2, 3, 1).reshape(n, Hc * Hc * N_DIRS)


def coarse_cell_xy(cell_xy, factor, H=GRID):
    """(H*H,2) 150-centres -> ((H/f)^2,2) averaged centres: the centroid of each block."""
    xy = cell_xy.view(1, H, H, 2).permute(0, 3, 1, 2)
    if factor > 1:
        xy = F.avg_pool2d(xy, kernel_size=factor, stride=factor)
    Hc = xy.shape[-1]
    return xy.permute(0, 2, 3, 1).reshape(Hc * Hc, 2)


def upsample_fields(x_flat, R, target=GRID):
    """Field (n, R^2) -> (n, target^2) by BILINEAR INTERPOLATION (align_corners=False).
    Identity when R==target. Pixel-centre convention: s = (o+0.5)*R/target - 0.5."""
    if R == target:
        return x_flat.float()
    n = x_flat.shape[0]
    x = x_flat.float().view(n, 1, R, R)
    x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    return x.reshape(n, -1)


def upsample_intensity(I_flat, R, target=GRID):
    """Intensity (n, R^2*16) -> (n, target^2*16), spatial bilinear (16 directions kept)."""
    if R == target:
        return I_flat.float()
    n = I_flat.shape[0]
    I = I_flat.float().view(n, R, R, N_DIRS).permute(0, 3, 1, 2)
    I = F.interpolate(I, size=(target, target), mode="bilinear", align_corners=False)
    return I.permute(0, 2, 3, 1).reshape(n, target * target * N_DIRS)


def resize_fields(x_flat, R_src, R_dst):
    """Field (n, R_src^2) -> (n, R_dst^2), in either direction.
    * R_dst < R_src: area mean (`adaptive_avg_pool2d`) -- exact when R_dst divides
      R_src (150->50 = 3x3 blocks), area-weighted otherwise (75->50).
    * R_dst > R_src: bilinear interpolation (25->50)."""
    if R_src == R_dst:
        return x_flat.float()
    n = x_flat.shape[0]
    x = x_flat.float().view(n, 1, R_src, R_src)
    x = (F.adaptive_avg_pool2d(x, R_dst) if R_dst < R_src
         else F.interpolate(x, size=(R_dst, R_dst), mode="bilinear", align_corners=False))
    return x.reshape(n, -1)


def resize_intensity(I_flat, R_src, R_dst):
    """Intensity (n, R_src^2*16) -> (n, R_dst^2*16), spatial only (16 directions kept)."""
    if R_src == R_dst:
        return I_flat.float()
    n = I_flat.shape[0]
    I = I_flat.float().view(n, R_src, R_src, N_DIRS).permute(0, 3, 1, 2)
    I = (F.adaptive_avg_pool2d(I, R_dst) if R_dst < R_src
         else F.interpolate(I, size=(R_dst, R_dst), mode="bilinear", align_corners=False))
    return I.permute(0, 2, 3, 1).reshape(n, R_dst * R_dst * N_DIRS)


def downsample_pred_intensity(I150_flat, R, H=GRID):
    """A 150 prediction (n, H^2*16) -> (n, R^2*16) by avg-pool (factor H/R). Used to
    bring a PCA/AE prediction (150 grid) back to the native resolution for the NATIVE
    table."""
    if R == H:
        return I150_flat.float()
    return avgpool_intensity(I150_flat, H // R, H=H)


def _resolution_assignment(N, res_list, seed):
    """Assign 1/3-1/3-1/3 of the snapshots to the resolutions (fixed-seed permutation)."""
    g = torch.Generator().manual_seed(seed)
    thirds = torch.tensor_split(torch.randperm(N, generator=g), len(res_list))
    res_of_snap = torch.empty(N, dtype=torch.long)
    for R, part in zip(sorted(res_list), thirds):
        res_of_snap[part] = R
    return res_of_snap


def build_mixed(clean_path=CLEAN_DATASET, out_path=MIXED_DATASET,
                assign_seed=0, split_seed=SPLIT_SEED, chunk=256, force=False):
    """NATIVE dataset grouped by resolution (for the Transformer-Based DeepONet),
    **TRAIN ONLY**: 1/3 at 150, 1/3 avg-pool to 75, 1/3 avg-pool to 25 (T, kappa, I;
    the 16 directions kept).

    The artefact holds ALL the snapshots (global-seed assignment), but only those of
    the TRAIN split (70 %) are used for training (cf. `fit_tbd`): val and test are always
    taken CLEAN at 150x150 through `load_clean_split` (input AND label). The val/test
    parts of the artefact only serve the degraded-input test."""
    if os.path.exists(out_path) and not force:
        print(f"[mixed] already present: {out_path} (force=True to rebuild)")
        return out_path
    print(f"[mixed] building from {clean_path}")
    ds = torch.load(clean_path, map_location="cpu", mmap=True, weights_only=False)["dict_data"]
    T_all, k_all, I_all, y_full = ds["u1_in"], ds["u2_in"], ds["I_target"], ds["y"]
    N = T_all.shape[0]
    cell_xy = y_full[::N_DIRS, :2].contiguous()
    phis16 = y_full[:N_DIRS, 2:3].contiguous()
    xy_lo, xy_hi = y_full[:, :2].min(0).values, y_full[:, :2].max(0).values
    res_of_snap = _resolution_assignment(N, RES_FACTORS, assign_seed)

    groups = {}
    for R in sorted(RES_FACTORS):
        f = RES_FACTORS[R]
        idx = torch.nonzero(res_of_snap == R, as_tuple=False).squeeze(1)
        n, Rc = idx.numel(), GRID // (RES_FACTORS[R])
        T_R = torch.empty(n, Rc * Rc); k_R = torch.empty(n, Rc * Rc)
        I_R = torch.empty(n, Rc * Rc * N_DIRS)
        for c0 in range(0, n, chunk):
            sl = idx[c0:c0 + chunk]; j = slice(c0, c0 + sl.numel())
            T_R[j] = avgpool_fields(T_all[sl].float(), f)
            k_R[j] = avgpool_fields(k_all[sl].float(), f)
            I_R[j] = avgpool_intensity(I_all[sl].float(), f)
        cxy = coarse_cell_xy(cell_xy, f)
        y3 = torch.cat([cxy.repeat_interleave(N_DIRS, 0), phis16.repeat(cxy.shape[0], 1)], dim=1)
        groups[R] = {"idx": idx, "T": T_R, "kappa": k_R, "I": I_R, "cell_xy": cxy,
                     "y3": y3, "factor": f, "grid": (Rc, Rc)}
        print(f"[mixed] res {R}×{R} : {n} snapshots")
    art = {"groups": groups, "res_of_snap": res_of_snap,
           "xy_lo": xy_lo, "xy_hi": xy_hi, "phis16": phis16,
           "meta": {"N": N, "H": GRID, "n_dirs": N_DIRS, "res_factors": RES_FACTORS,
                    "assign_seed": assign_seed, "split_seed": split_seed,
                    "split": "70/15/15", "downsample": "avg_pool"}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(art, out_path)
    print(f"[mixed] -> {out_path} ({os.path.getsize(out_path)/1024**3:.2f} GB)")
    return out_path


def build_interp150(mixed_path=MIXED_DATASET, clean_path=CLEAN_DATASET,
                    out_path=INTERP150_DATASET, chunk=256, force=False):
    """150x150 INTERPOLATED dataset (for PCA/AE, fixed dimension): the 75/25 thirds are
    bilinearly upsampled to 150 (T, kappa, I). A drop-in for `deeponet` (same keys,
    unchanged seed-42 split). Saves and returns the path."""
    if os.path.exists(out_path) and not force:
        print(f"[interp150] already present: {out_path} (force=True to rebuild)")
        return out_path
    print(f"[interp150] building from {mixed_path} + {clean_path}")
    groups = torch.load(mixed_path, map_location="cpu", weights_only=False)["groups"]
    ds = torch.load(clean_path, map_location="cpu", weights_only=False)
    dd = ds["dict_data"]
    T_all, k_all, I_all = dd["u1_in"], dd["u2_in"], dd["I_target"]
    extra = [(ds[dk]["u1_in"], ds[dk]["u2_in"]) for dk in ("dict_pde", "dict_bc")
             if ds[dk]["u1_in"].data_ptr() != T_all.data_ptr()]
    for R in sorted(groups):
        if R == GRID:
            continue
        gr = groups[R]; idx = gr["idx"]; n = idx.numel()
        print(f"[interp150] res {R}->150 : {n} snapshots")
        for c0 in range(0, n, chunk):
            sl = idx[c0:c0 + chunk]; j = slice(c0, c0 + sl.numel())
            T150 = upsample_fields(gr["T"][j], R); k150 = upsample_fields(gr["kappa"][j], R)
            T_all[sl] = T150; k_all[sl] = k150; I_all[sl] = upsample_intensity(gr["I"][j], R)
            for (u1e, u2e) in extra:
                u1e[sl] = T150; u2e[sl] = k150
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(ds, out_path)
    print(f"[interp150] -> {out_path} ({os.path.getsize(out_path)/1024**3:.2f} GB)")
    return out_path


def fixed_res_dataset_path(R):
    return os.path.join(DERIVED_DIR, f"fixedres{R}_dataset_phi.pt")


def build_fixed_res(R, mixed_path=MIXED_DATASET, clean_path=CLEAN_DATASET,
                    out_path=None, chunk=256, force=False):
    """FIXED-resolution R x R dataset for the fixed-input-dimension models (PCA/AE).

    Scenario: the data arrives at heterogeneous resolutions; a FIXED-dimension model has
    to bring everything to ONE working resolution R. So EVERYTHING (T, kappa AND I) is
    brought there:
        150^2 --area mean--> R^2 ,  75^2 --area mean--> R^2 ,  25^2 --bilinear--> R^2
    For R=50 the 150^2 is DEGRADED (the high-resolution information is lost) -- exactly
    what the Transformer-Based DeepONet does not need to do (it ingests the native resolution).

    * TRAIN snapshots: from their native resolution (mixed artefact) -> R^2.
    * VAL snapshots: from the CLEAN 150^2 solver -> R^2 (that is all the model can see
      as input); the final test still compares against the 150^2 truth (the trunk is
      coordinate-based, cf. evaluate.py).

    Sortie : drop-in `deeponet` (dict_pde/bc/data, grille R², split seed-42)."""
    out_path = out_path or fixed_res_dataset_path(R)
    if os.path.exists(out_path) and not force:
        print(f"[fixed{R}] already present: {out_path}")
        return out_path
    print(f"[fixed{R}] building (train from {os.path.basename(mixed_path)}, clean val)")
    art = torch.load(mixed_path, map_location="cpu", weights_only=False)
    groups = art["groups"]
    ds = torch.load(clean_path, map_location="cpu", weights_only=False)
    dd = ds["dict_data"]
    N = dd["u1_in"].shape[0]
    is_train = split_mask(N, "train", art["meta"]["split_seed"])

    nc = R * R
    T_new = torch.empty(N, nc); k_new = torch.empty(N, nc)
    I_new = torch.empty(N, nc * N_DIRS)

    for Rsrc in sorted(groups):
        gr = groups[Rsrc]
        loc = is_train[gr["idx"]].nonzero(as_tuple=False).squeeze(1)
        print(f"[fixed{R}]   train {Rsrc}² -> {R}² : {loc.numel()} snapshots")
        for c0 in range(0, loc.numel(), chunk):
            sl = loc[c0:c0 + chunk]
            tgt = gr["idx"][sl]
            T_new[tgt] = resize_fields(gr["T"][sl], Rsrc, R)
            k_new[tgt] = resize_fields(gr["kappa"][sl], Rsrc, R)
            I_new[tgt] = resize_intensity(gr["I"][sl], Rsrc, R)

    held_idx = (~is_train).nonzero(as_tuple=False).squeeze(1)
    print(f"[fixed{R}]   val+test (propres) 150² -> {R}² : {held_idx.numel()} snapshots")
    for c0 in range(0, held_idx.numel(), chunk):
        sl = held_idx[c0:c0 + chunk]
        T_new[sl] = resize_fields(dd["u1_in"][sl], GRID, R)
        k_new[sl] = resize_fields(dd["u2_in"][sl], GRID, R)
        I_new[sl] = resize_intensity(dd["I_target"][sl], GRID, R)

    y150 = dd["y"]
    cxy = coarse_cell_xy(y150[::N_DIRS, :2].contiguous(), GRID // R)
    phis16 = y150[:N_DIRS, 2:3].contiguous()
    y_R = torch.cat([cxy.repeat_interleave(N_DIRS, 0), phis16.repeat(nc, 1)], dim=1)

    for dk in ("dict_pde", "dict_bc"):
        ds[dk]["u1_in"] = T_new
        ds[dk]["u2_in"] = k_new
    dd["u1_in"], dd["u2_in"], dd["I_target"], dd["y"] = T_new, k_new, I_new, y_R

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(ds, out_path)
    print(f"[fixed{R}] -> {out_path} ({os.path.getsize(out_path)/1024**3:.2f} GB)")
    return out_path


def load_mixed(path=MIXED_DATASET, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def load_clean_split(which="test", clean_path=CLEAN_DATASET, split_seed=SPLIT_SEED, limit=0):
    """One split at FULL RESOLUTION: (T, kappa, I, y150, idx).
    Input AND label at 150x150, never degraded -- identical for every model.

    `val` is used for monitoring and best-checkpoint selection during training, `test`
    for EVERYTHING that is reported."""
    dd = torch.load(clean_path, map_location="cpu", mmap=True, weights_only=False)["dict_data"]
    N = dd["u1_in"].shape[0]
    idx = split_indices(N, split_seed)[which]
    if limit:
        idx = idx[:limit]
    return (dd["u1_in"][idx], dd["u2_in"][idx], dd["I_target"][idx],
            dd["y"].float(), idx)


def split_indices(N, seed=SPLIT_SEED, train_frac=0.70, val_frac=0.15):
    """The project's train / val / test indices -- the SAME arithmetic as
    `utils.load_and_split_dataset`: consecutive slices of a single
    `randperm(N)` sous `manual_seed(seed)`, 70 / 15 / 15.

    Reimplemented here (rather than imported) to avoid the `utils` name collision
    between the sub-package and the parent; any divergence from the parent would be a
    silent bug, hence the literal duplication of the arithmetic."""
    torch.manual_seed(seed)
    perm = torch.randperm(N)
    n_tr, n_va = int(train_frac * N), int(val_frac * N)
    return {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}


def split_mask(N, which="train", seed=SPLIT_SEED):
    """Boolean (N,) mask of the requested split -- handy to filter the resolution
    groups, which are indexed by original snapshot."""
    m = torch.zeros(N, dtype=torch.bool)
    m[split_indices(N, seed)[which]] = True
    return m


def load_and_split(path, seed=SPLIT_SEED, map_location="cpu"):
    """Load a drop-in dataset (dict_pde/bc/data) and split it into train/val/test
    EXACTLY like the parent's `load_and_split_dataset` (same seed -> same snapshots as
    the Transformer-Based DeepONet). Same return structure: `data` = (train, val), test
    separate."""
    ds = torch.load(path, map_location=map_location, weights_only=False)
    dict_pde, dict_bc, dict_data = ds["dict_pde"], ds["dict_bc"], ds["dict_data"]
    n = dict_data["u1_in"].shape[0]
    sp = split_indices(n, seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    y_full = dict_data["y"]

    def _split(d):
        a, b, c = {}, {}, {}
        for k in list(d.keys()):
            v = d.pop(k)
            if k == "y":
                a[k] = b[k] = c[k] = v
            else:
                a[k], b[k], c[k] = v[tr], v[va], v[te]
        return a, b, c

    pde, bc, data = _split(dict_pde), _split(dict_bc), _split(dict_data)
    return {"pde": pde[:2], "bc": bc[:2], "data": data[:2],
            "pde_test": pde[2], "bc_test": bc[2], "data_test": data[2],
            "y_full": y_full, "train_idx": tr, "val_idx": va, "test_idx": te}


def tbd_encode_native(encoder, T_R, k_R, R, device, chunk=256):
    """Transformer-Based DeepONet: encode (T, kappa) at the NATIVE resolution R x R ->
    flattened latent (n, n_latent*dim). Per-channel standardisation (stats of the
    pretrained encoder),
    bypassing the fixed grid."""
    mean, std = encoder.stats
    mean, std = mean.to(device), std.to(device)
    outs = []
    encoder.model.eval()
    with torch.no_grad():
        for i in range(0, T_R.shape[0], chunk):
            a = T_R[i:i + chunk].float().view(-1, 1, R, R)
            b = k_R[i:i + chunk].float().view(-1, 1, R, R)
            x = (torch.cat([a, b], 1).to(device) - mean) / std
            outs.append(encoder.model.encode(x).cpu())
    return torch.cat(outs, 0).float()


def compute_tbd_latents(encoder, groups, device):
    """latents[R] = (n, latent_dim) for each group (frozen encoder)."""
    lat = {}
    for R, gr in groups.items():
        lat[R] = tbd_encode_native(encoder, gr["T"], gr["kappa"], R, device)
        print(f"  [latents Transformer-Based DeepONet] res {R}×{R} : "
              f"{lat[R].shape[0]} snapshots")
    return lat


def build_tbd_operator(encoder, I_scale, xy_lo, xy_hi, device, freeze_encoder=False):
    """Instantiate a TBDOperator (pretrained Transformer-Based DeepONet encoder + dec_I
    coordinate decoder).

    `freeze_encoder=False` (the default, and what the paper describes) leaves the
    encoder trainable, so the operator stage fine-tunes it exactly as
    `tbdeeponet.fit_operator` does on the fixed-resolution dataset."""
    from tbdeeponet import TBDOperator
    op = TBDOperator(encoder.model, I_scale=I_scale, dec_depth=encoder.dec_depth,
                     heads=encoder.heads, mlp_ratio=encoder.mlp_ratio,
                     n_fourier=encoder.n_fourier, fourier_sigma=encoder.fourier_sigma).to(device)
    op.xy_lo.copy_(xy_lo.to(device)); op.xy_hi.copy_(xy_hi.to(device))
    op.ae.requires_grad_(not freeze_encoder)
    return op


def median_I_scale(groups, is_train, cap=int(2e7)):
    """Median of I>1e-3 over the TRAIN snapshots (bounded subsample)."""
    vals, per = [], max(1, cap // max(1, len(groups)))
    for R, gr in groups.items():
        loc = is_train[gr["idx"]].nonzero(as_tuple=False).squeeze(1)
        if loc.numel() == 0:
            continue
        I = gr["I"][loc].reshape(-1); I = I[I > 1e-3]
        if I.numel() > per:
            I = I[torch.randint(0, I.numel(), (per,))]
        vals.append(I)
    return float(torch.cat(vals).median()) if vals else 1.0


def fit_tbd(operator, latents, groups, is_train, val_pack, I_scale, device, epochs=1000,
            batch_size=64, n_query=8192, lr=1e-3, gamma=0.9977, val_every=25,
            log_every=10, grad_clip=1.0, save_fn=None, seed=0, data_loss="rel_l2",
            phys_val=None, monitor_phys=True, n_query_phys=2048, history_name=None,
            finetune_encoder=True, encoder_lr=1e-4, stats=None):
    """Train dec_I on the mixed native data: each step = one group, a batch of its
    TRAIN snapshots (filtered by `is_train` -- val and test are never seen), n_query
    coords drawn from y3_R.

    `finetune_encoder` (DEFAULT, and what Appendix A describes): the encoder is trained
    jointly with dec_I at `encoder_lr`, under the same decay -- the regime
    `tbdeeponet.fit_operator` uses on the fixed-resolution dataset. The input pair of
    each batch then goes through the encoder INSIDE the graph, at its own native
    resolution, so `latents` is ignored and may be None. Set it to False to precompute
    the latents once and train dec_I alone (faster, but no longer what the paper says).
    `stats` = the encoder's per-channel (mean, std); required when fine-tuning.

    `data_loss`: "rel_l2" (DEFAULT, like `tbdeeponet.fit_operator`) = PER-SNAPSHOT relative L2,
    so every snapshot weighs the same whatever its intensity -- and the training
    criterion coincides with the reported metric. "mse" = MSE(I)/I_scale^2, the older
    behaviour, dominated by the brightest snapshots.

    `val_pack` = (z_val, I_val, coords_val): validation at FULL 150x150 RESOLUTION
    (precomputed latents, frozen encoder) -- so best-model selection happens on CLEAN,
    never degraded, data.

    `history_name` writes `checkpoints/<name>_history.json` and its learning-curve
    figure, through the very `utils.TrainingHistory` / `utils.plot_learning_curves` the
    PCA/AE path uses (`deeponet.Trainer`): same series names, same panels, same cadence,
    so both families are readable on the same axes.

    `monitor_phys` (needs `phys_val` = (T_val, kappa_val, y150)) MEASURES the RTE
    residual without ever optimising it, at the `val_every` cadence, with the collocation
    and the `utils.rel_phys_loss` normalisation of `tbdeeponet.fit_operator`. TRAIN side the
    residual is averaged over the resolution groups -- every model is measured on ITS OWN
    (degraded) training data, which is the whole point of the experiment; the VAL side,
    on the clean 150^2 split, is the number comparable across models."""
    if data_loss not in ("mse", "rel_l2"):
        raise ValueError(f"data_loss inconnu : {data_loss!r} (mse | rel_l2)")
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    scale_sq = I_scale ** 2

    def data_loss_fn(pred, target):
        if data_loss == "rel_l2":
            return ((pred - target).norm(dim=1) / target.norm(dim=1).clamp_min(1e-8)).mean()
        return F.mse_loss(pred, target) / scale_sq

    if finetune_encoder and stats is None:
        raise ValueError("fit_tbd(finetune_encoder=True) needs `stats` "
                         "(the encoder's per-channel mean/std)")
    dev_lat, coords, loc_tr = {}, {}, {}
    for R, gr in groups.items():
        if not finetune_encoder:
            dev_lat[R] = latents[R].to(device)
        coords[R] = gr["y3"].float().to(device)
        loc_tr[R] = is_train[gr["idx"]].nonzero(as_tuple=False).squeeze(1)
    val_fields, I_val_sub, coords_val = val_pack
    coords_val = coords_val.to(device)
    N_LAT, DIM = operator.n_latent, operator.dim
    mean = std = None
    if stats is not None:
        mean, std = (t.to(device) for t in stats)

    def _tokens(R, bl, grad):
        """Latents (b, N, D) of a batch of the resolution-R group. Fine-tuning: the
        (T, kappa) pair is standardised and encoded HERE, at its native R x R, so the
        encoder sits inside the graph. Frozen: read off the precomputed table."""
        if not finetune_encoder:
            z = dev_lat[R][bl.to(device)]
            return z.view(z.shape[0], N_LAT, DIM)
        x = torch.cat([groups[R]["T"][bl].float().view(-1, 1, R, R),
                       groups[R]["kappa"][bl].float().view(-1, 1, R, R)], 1).to(device)
        x = (x - mean) / std
        with contextlib.nullcontext() if grad else torch.no_grad():
            return operator.encode_tokens(x)

    def _val_tokens():
        """Latents of the CLEAN 150^2 val block, re-encoded whenever the encoder moved."""
        outs = []
        with torch.no_grad():
            for b in range(0, val_fields.shape[0], batch_size):
                outs.append(operator.encode_tokens(val_fields[b:b + batch_size].to(device)))
        return torch.cat(outs, 0)

    if finetune_encoder:
        dec_params = list(operator.dec_I.parameters())
        enc_params = list(operator.ae.parameters())
        opt = optim.Adam([{"params": dec_params, "lr": lr},
                          {"params": enc_params, "lr": encoder_lr}])
        print(f"[Transformer-Based DeepONet] encoder FINE-TUNED: lr dec_I={lr:g}, "
              f"lr encoder={encoder_lr:g}, same gamma={gamma}")
    else:
        opt = optim.Adam(operator.parameters(), lr=lr)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    res_list = sorted(groups)

    hist = TrainingHistory(history_name or "res-TBD", save_dir=CKPT_DIR,
                           meta={"w_data": 1.0, "w_phys": 0.0, "w_bc": 0.0, "epochs": epochs,
                                 "loss_balancing": "static",
                                 "finetune_encoder": bool(finetune_encoder),
                                 "encoder_lr": encoder_lr if finetune_encoder else None,
                                 "data_loss": data_loss})

    monitor_phys = monitor_phys and phys_val is not None
    if monitor_phys:
        from deeponet import IB_COEF
        T_val_raw, kappa_val_raw, y_val_full = phys_val
        coords_phys_val = y_val_full.float().to(device)

        def _phys_measure(z_b, T_raw, kap_raw, snap, coords_all):
            """(relative, absolute) RTE residual on one batch, create_graph=False: the
            source is rebuilt at the collocation cell, the derivatives are exact
            (autodiff through dec_I), nothing survives the call."""
            snap = snap.cpu()
            pidx = torch.randint(0, coords_all.shape[0], (n_query_phys,), device=device)
            c_idx = (pidx // N_DIRS).cpu()
            T_at = T_raw[snap][:, c_idx].float().to(device)
            kap_at = kap_raw[snap][:, c_idx].float().to(device)
            u1_eval = kap_at * IB_COEF * T_at ** 4
            u2_eval = -kap_at
            residual, I_pde = operator.pde_residual([z_b, None], coords_all[pidx],
                                                    u1_eval, u2_eval, create_graph=False)
            rel, mse, _ = rel_phys_loss(residual, I_pde, u1_eval, u2_eval, return_parts=True)
            return rel.item(), mse.item()

    print(f"[Transformer-Based DeepONet] {sum(loc_tr[R].numel() for R in res_list)} "
          f"train snapshots (degraded), "
          f"val PROPRE {val_fields.shape[0]} snapshots 150², loss={data_loss}, "
          f"I_scale={I_scale:.3e}, M={n_query}, grad_clip={grad_clip}, val/{val_every} ep."
          f"{f', phys monitored on {n_query_phys} pts' if monitor_phys else ''}")

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        operator.train()
        if not finetune_encoder:
            operator.ae.eval()
        steps = []
        for R in res_list:
            tr = loc_tr[R][torch.randperm(loc_tr[R].numel())]
            steps += [(R, tr[b:b + batch_size]) for b in range(0, tr.numel(), batch_size)]
        ep_loss = 0.0
        for si in torch.randperm(len(steps)).tolist():
            R, bl = steps[si]
            cidx = torch.randint(0, coords[R].shape[0], (n_query,), device=device)
            target = groups[R]["I"][bl][:, cidx.cpu()].to(device)
            opt.zero_grad()
            pred = operator.predict_I(_tokens(R, bl, grad=True), coords[R][cidx])
            loss = data_loss_fn(pred, target)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(operator.parameters(), grad_clip)
            opt.step(); ep_loss += loss.item()
        sched.step()
        train_loss = ep_loss / max(1, len(steps))

        do_val = (ep + 1) % val_every == 0 or ep in (0, epochs - 1)
        phys_train_mon = phys_val_mon = None
        phys_abs_train_mon = phys_abs_val_mon = None
        if do_val:
            z_val = _val_tokens().flatten(1)
            v = _tbd_val(operator, z_val, I_val_sub, coords_val, device)
            if monitor_phys:
                rels, abss = [], []
                for R in res_list:
                    sel = loc_tr[R][torch.randperm(loc_tr[R].numel())[:batch_size]]
                    r, a = _phys_measure(_tokens(R, sel, grad=False).flatten(1),
                                         groups[R]["T"], groups[R]["kappa"], sel, coords[R])
                    rels.append(r); abss.append(a)
                phys_train_mon = sum(rels) / len(rels)
                phys_abs_train_mon = sum(abss) / len(abss)
                nv = min(batch_size, z_val.shape[0])
                phys_val_mon, phys_abs_val_mon = _phys_measure(
                    z_val[:nv], T_val_raw, kappa_val_raw, torch.arange(nv), coords_phys_val)
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().cpu().clone() for k, t in operator.state_dict().items()}
                if save_fn is not None:
                    save_fn(operator, {"best_val_I_l2": best_val, "epoch": ep + 1})

        hist.log(ep + 1, train_total=train_loss, train_data=train_loss,
                 train_phys=phys_train_mon, train_phys_abs=phys_abs_train_mon,
                 val_data=(v if do_val else None), val_phys=phys_val_mon,
                 val_phys_abs=phys_abs_val_mon)

        if (ep + 1) % log_every == 0 or ep == 0:
            vt = f"{v:.4f}" if do_val else "  --  "
            print(f"  [Transformer-Based DeepONet] ep {ep+1:4d}/{epochs} "
                  f"| train {train_loss:.5f} "
                  f"| val I-L2 {vt} | best {best_val:.4f}")
            if phys_train_mon is not None:
                print(f"           | phys {phys_train_mon:.6f} (monitored only) "
                      f"| val phys {phys_val_mon:.6f}")
    if best_state is not None:
        operator.load_state_dict(best_state)
        print(f"  [Transformer-Based DeepONet] best reloaded (val I-L2 = {best_val:.4f})")
    if history_name is not None:
        hist.meta["best_val"] = best_val
        hist.save()
        try:
            plot_learning_curves(history_name, save_dir=CKPT_DIR,
                                 title=f"{history_name} — learning curves")
        except Exception as e:
            print(f"  [!] learning-curve plot skipped: {e}")
    operator.eval()
    return operator, best_val


@torch.no_grad()
def _tbd_val(operator, z_val, I_val_sub, coords_val, device, batch=64):
    """Validation on the CLEAN 150x150 split (fixed coords): mean relative I-L2."""
    operator.eval()
    ratios = []
    for b in range(0, z_val.shape[0], batch):
        sl = slice(b, b + batch)
        pred = operator([z_val[sl], None], coords_val)
        ratios.append(rel_l2_rows(pred, I_val_sub[sl].to(device)).cpu())
    return float(torch.cat(ratios).mean()) if ratios else float("nan")


def make_tbd_val_pack(encoder, T_val, k_val, I_val, y150, device, n_query=8192, seed=0):
    """Prepare the Transformer-Based DeepONet's CLEAN validation: the STANDARDISED 150^2
    input pair, plus a FIXED (deterministic) subset of coords so the monitoring is
    comparable epoch to epoch.

    The fields are returned raw rather than encoded: the encoder moves during training
    (fine-tuning), so `fit_tbd` re-encodes them at every validation -- selecting the best
    checkpoint on latents produced by an older encoder would score the wrong pair. Kept
    on CPU (900 x 2 x 150^2 = 160 MB), moved batch by batch."""
    mean, std = (t.cpu() for t in encoder.stats)
    fields = (torch.cat([T_val.float().view(-1, 1, GRID, GRID),
                         k_val.float().view(-1, 1, GRID, GRID)], 1) - mean) / std
    gen = torch.Generator().manual_seed(seed)
    cidx = torch.randint(0, y150.shape[0], (min(2 * n_query, y150.shape[0]),), generator=gen)
    return fields, I_val[:, cidx], y150[cidx]


def rel_l2_rows(pred, true):
    """Row-wise rel-L2: ||pred-true||/||true||, (b,)."""
    num = (pred - true).pow(2).sum(1).sqrt()
    den = true.pow(2).sum(1).sqrt().clamp_min(1e-12)
    return num / den


def to_G(I, nc):
    """(b, nc*16) -> (b, nc) : radiation incidente G = ∑_dir I * 4*pi/N_DIRS,
    le poids d'angle solide du solveur (cf. utils.dir_weight)."""
    return I.view(I.shape[0], nc, N_DIRS).sum(2) * (4.0 * math.pi / N_DIRS)


def tbd_ckpt(name):
    return os.path.join(CKPT_DIR, f"{name}.ptop")


def modular_ckpt(name):
    return os.path.join(CKPT_DIR, f"{name}.pth")


def save_tbd(operator, encoder, name, meta=None):
    from tbdeeponet import save_operator
    os.makedirs(CKPT_DIR, exist_ok=True)
    save_operator(operator, encoder, tbd_ckpt(name), meta=meta or {})


def load_tbd(name, device):
    """(operator, encoder_wrapper) of a native Transformer-Based DeepONet."""
    from tbdeeponet import load_operator
    return load_operator(tbd_ckpt(name), device=device)


def load_modular(name, device):
    """Load a standard MultiBranchDeepONet (architecture AND head inferred from the
    .pth, cf. `deeponet.load_model`)."""
    from deeponet import load_model
    return load_model(modular_ckpt(name), device)


def load_encoder_for(family, device, res=GRID, name=None):
    """Encoder of the requested family, at the `res` working resolution.

    Always an encoder of THIS sub-package, fitted on the mixed-resolution train split
    (`fit_tbd_encoder_on_mixed` / `fit_encoder_at_res`) -- never one of the main
    campaign's, which saw the clean 150^2 fields.

    `name` is the RUN name, and it matters: an AE encoder fine-tuned with the branch and
    trunk nets is saved as `encoder_autoencoder_<run>_ft.pt`, and pairing the operator
    back with the pretrained encoder instead would evaluate two halves that never met
    (~3x the error). Given a name, the fine-tuned encoder wins whenever it exists."""
    if family == "tbd":
        from tbdeeponet import TBDEncoder
        return TBDEncoder.load(ENC_TBD, device=device)
    from encoders import load_encoder, encoder_path
    kind = "pca" if family == "pca" else "autoencoder"
    path = None
    if name is not None:
        ft = encoder_path(f"{name}_ft", kind, save_dir=CKPT_DIR)
        if os.path.exists(ft):
            print(f"[{name}] fine-tuned encoder: {os.path.basename(ft)}")
            path = ft
    return load_encoder(kind, path or res_encoder_path(family, res), device=device)


def res_encoder_path(family, R, save_dir=None):
    """Path of an encoder fitted at resolution R (inside the sub-package, without
    the project's 150^2 encoders)."""
    kind = "pca" if family == "pca" else "autoencoder"
    ext = "pkl" if family == "pca" else "pt"
    return os.path.join(save_dir or CKPT_DIR, f"encoder_{kind}_rob{R}.{ext}")


def fit_encoder_at_res(family, R, dataset_path, device, latent_dim=500, ae_epochs=1000,
                       ae_batch=64, ae_lr=1e-3, pca_components=500, pca_scaler=True,
                       force=False):
    """Fit (PCA) / train (AE) an encoder on the R^2 fields of the TRAIN SPLIT.

    `dataset_path` is the mixed-resolution dataset brought to this model's working grid
    (interp150 for R = 150, fixedres50 for R = 50), so the encoder sees exactly the
    fields the operator will be trained on.

    Same hyperparameters as the main campaign (PCA: 500 axes + StandardScaler; AE:
    latent 500, 1000 epochs at lr 1e-3) -- only the data changes."""
    from encoders import make_encoder
    path = res_encoder_path(family, R)
    if os.path.exists(path) and not force:
        print(f"[encoder {family}@{R}] already present: {path}")
        return path
    splits = load_and_split(dataset_path)
    data_tr, data_va = splits["data"]
    cfg = dict(n_components=pca_components, use_scaler=pca_scaler, latent_dim=latent_dim,
               grid_shape=(R, R), epochs=ae_epochs, batch_size=ae_batch, lr=ae_lr,
               device=device)
    kind = "pca" if family == "pca" else "autoencoder"
    enc = make_encoder(kind, **cfg)
    print(f"[encoder {family}@{R}] fitting on {data_tr['u1_in'].shape[0]} snapshots at {R}^2")
    enc.fit_transform_dicts({"u1_in": data_tr["u1_in"], "u2_in": data_tr["u2_in"]},
                            ("u1_in", "u2_in"),
                            dict_val={"u1_in": data_va["u1_in"], "u2_in": data_va["u2_in"]})
    os.makedirs(CKPT_DIR, exist_ok=True)
    enc.save(path)
    return path


def fit_tbd_encoder_on_mixed(device, path=ENC_TBD, epochs=1000, batch_size=64, lr=1e-3,
                             n_query=4096, patch=5, dim=128, n_latent=4, depth=4,
                             dec_depth=4, heads=8, mlp_ratio=2.0, n_fourier=64,
                             fourier_sigma=3.0, force=False, log_every=10):
    """Stage one of the Transformer-Based DeepONet: pretrain the encoder+decoder on the
    RECONSTRUCTION of the input pair, ON THE MIXED-RESOLUTION DATASET.

    Each step draws one resolution group (150/75/25), reads its (T, kappa) at that
    native resolution and reconstructs the very same field at `n_query` coordinates
    drawn on its own grid -- so a snapshot presented at 25^2 is never compared with
    anything it does not carry. On top of that the input is randomly subsampled (the
    FunDiff augmentation, cf. `tbdeeponet.valid_subsample_factors`), which is what makes
    the patch grid vary within a single resolution group too.

    Only the TRAIN snapshots are ever read; the best checkpoint is picked on the CLEAN
    150^2 val block, the same criterion `fit_tbd` uses for the operator. `patch = 5`
    (not the 10 of the main campaign) is what lets a 25^2 input still be cut into 5x5
    tokens.

    Saved in the `TBDEncoder` format, so `TBDEncoder.load` and `tbd_encode_native` read
    it back unchanged."""
    from tbdeeponet import TBDEncoder, valid_subsample_factors
    if os.path.exists(path) and not force:
        print(f"[encoder tbd@mixed] already present: {path}")
        return path

    build_mixed()
    art = load_mixed()
    groups = art["groups"]
    is_train = split_mask(art["meta"]["N"], "train", art["meta"]["split_seed"])
    loc_tr = {R: is_train[gr["idx"]].nonzero(as_tuple=False).squeeze(1)
              for R, gr in groups.items()}
    res_list = sorted(groups)

    enc = TBDEncoder(grid_shape=(GRID, GRID), epochs=epochs, batch_size=batch_size, lr=lr,
                     device=device, patch=patch, dim=dim, n_latent=n_latent, depth=depth,
                     dec_depth=dec_depth, heads=heads, mlp_ratio=mlp_ratio,
                     n_fourier=n_fourier, fourier_sigma=fourier_sigma, n_query=n_query)

    # Per-channel standardisation pooled over every TRAIN snapshot of every group: the
    # groups hold different numbers of cells, so the moments are accumulated by hand
    # rather than averaged group by group.
    s1 = torch.zeros(2, dtype=torch.float64)
    s2 = torch.zeros(2, dtype=torch.float64)
    cnt = 0
    for R in res_list:
        gr, sel = groups[R], loc_tr[R]
        for c, key in enumerate(("T", "kappa")):
            v = gr[key][sel].double()
            s1[c] += v.sum(); s2[c] += (v * v).sum()
        cnt += sel.numel() * R * R
    mean = (s1 / cnt).float().view(1, 2, 1, 1)
    std = ((s2 / cnt - (s1 / cnt) ** 2).clamp_min(1e-16).sqrt()).float().view(1, 2, 1, 1)
    enc.stats = (mean.to(device), std.to(device))
    print(f"[encoder tbd@mixed] per-channel stats on {cnt:,} values: "
          f"mean {mean.flatten().tolist()}, std {std.flatten().tolist()}")

    coords = {R: TBDEncoder._grid_coords((R, R)).to(device) for R in res_list}
    factors = {R: valid_subsample_factors((R, R), patch) for R in res_list}
    print(f"[encoder tbd@mixed] random input subsampling per group: "
          + ", ".join(f"{R}²->{factors[R]}" for R in res_list))

    model = enc._make_model().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    T_va, k_va, _, _, _ = load_clean_split("val")
    coords_val = coords[GRID]
    gen = torch.Generator().manual_seed(0)
    val_idx = torch.randint(0, GRID * GRID, (min(2 * n_query, GRID * GRID),), generator=gen).to(device)
    val_fields = ((torch.cat([T_va.float().view(-1, 1, GRID, GRID),
                              k_va.float().view(-1, 1, GRID, GRID)], 1) - mean) / std)

    hist = TrainingHistory("res-TBD-encoder", save_dir=CKPT_DIR,
                           meta={"stage": "reconstruction", "dataset": "mixed 150/75/25",
                                 "epochs": epochs, "patch": patch,
                                 "subsample": {str(R): list(factors[R]) for R in res_list}})
    rpath = os.path.join(CKPT_DIR, "res-TBD-encoder_enc_resume.pt")
    start_epoch, best_val, best_state = 0, float("inf"), None
    if os.path.exists(rpath):
        st = torch.load(rpath, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        start_epoch, best_val, best_state = st["epoch"], st["best_val"], st["best_state"]
        hist.series = st.get("hist", hist.series)
        print(f"[encoder tbd@mixed] RESUMING at epoch {start_epoch}/{epochs} "
              f"(best val MSE = {best_val:.6f})")

    n_tr = sum(loc_tr[R].numel() for R in res_list)
    print(f"[encoder tbd@mixed] {n_tr} train snapshots, val on {val_fields.shape[0]} "
          f"clean 150² ones, M={n_query} coords/step")

    for ep in range(start_epoch, epochs):
        model.train()
        steps = []
        for R in res_list:
            tr = loc_tr[R][torch.randperm(loc_tr[R].numel())]
            steps += [(R, tr[b:b + batch_size]) for b in range(0, tr.numel(), batch_size)]
        ep_loss = 0.0
        for si in torch.randperm(len(steps)).tolist():
            R, bl = steps[si]
            x = ((torch.cat([groups[R]["T"][bl].float().view(-1, 1, R, R),
                             groups[R]["kappa"][bl].float().view(-1, 1, R, R)], 1)
                  ).to(device) - mean.to(device)) / std.to(device)
            s = factors[R][torch.randint(len(factors[R]), (1,)).item()]
            enc_in = x[:, :, ::s, ::s] if s > 1 else x
            idx = torch.randint(0, R * R, (min(n_query, R * R),), device=device)
            target = x.view(x.shape[0], 2, -1)[:, :, idx].permute(0, 2, 1)
            opt.zero_grad()
            pred, _ = model(enc_in, coords[R][idx])
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        train_mse = ep_loss / max(1, len(steps))

        model.eval()
        with torch.no_grad():
            v, nb = 0.0, 0
            for i in range(0, val_fields.shape[0], batch_size):
                vb = val_fields[i:i + batch_size].to(device)
                tgt = vb.view(vb.shape[0], 2, -1)[:, :, val_idx].permute(0, 2, 1)
                pv, _ = model(vb, coords_val[val_idx])
                v += F.mse_loss(pv, tgt).item(); nb += 1
            val_mse = v / max(1, nb)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

        hist.log(ep + 1, train_total=train_mse, train_data=train_mse, val_data=val_mse)
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"  [encoder tbd@mixed] ep {ep+1:4d}/{epochs} | train MSE {train_mse:.6f} "
                  f"| val MSE {val_mse:.6f} | best {best_val:.6f}", flush=True)
        if (ep + 1) % 25 == 0:
            save_atomic({"epoch": ep + 1, "model": model.state_dict(),
                         "optimizer": opt.state_dict(), "best_val": best_val,
                         "best_state": best_state, "hist": hist.series}, rpath)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[encoder tbd@mixed] best reloaded (val MSE = {best_val:.6f})")
    model.eval()
    enc.model = model
    enc.fitted = True
    enc.save(path)
    hist.meta["best_val"] = best_val
    hist.save()
    try:
        plot_learning_curves("res-TBD-encoder", save_dir=CKPT_DIR,
                             title="res-TBD encoder — reconstruction on the mixed data")
    except Exception as e:
        print(f"  [!] learning-curve plot skipped: {e}")
    if os.path.exists(rpath):
        os.remove(rpath)
    return path
