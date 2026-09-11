#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_LB = os.path.join(_SRC, "lower_bound")
for _p in (_SRC, _LB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from deeponet import MLP, INPUT_KEYS
from tbdeeponet import (TBDOperator, FunctionAutoencoder, TBDEncoder,
                 RandomFourierEmbedding, fit_operator, save_operator)
from utils import (DEFAULT_DATASET, Evaluator, load_and_split_dataset, save_figure,
                   solver_h5, split_data)

CKPT = os.path.join(_HERE, "checkpoints")
CACHE = os.path.join(_HERE, "cache")
SETS = {"train": "TRAIN", "test": "TEST"}
REFERENCES = ("abl-dot-branch-affine", "abl-dot-branch", "abl-dot-branch-raw",
              "abl-dot", "abl-dot-notrunk", "TBD-N1-D128",
              "PCA-p128-softplus", "AE-p128-softplus")


def operator_path(name):
    return os.path.join(CKPT, f"operator_dot_{name}.pt")


class DotDecoder(nn.Module):
    """Branch.trunk junction: (z_tokens, coords) -> <flatten(z), t(coords)> + b.

    `coords` is (M, 3), shared across the batch, or (B, M, 3)."""

    def __init__(self, p, dim=128, hidden=(512, 512, 512, 512), n_fourier=64,
                 fourier_sigma=3.0, coord_dim=3, rff=True, latent_dim=None,
                 branch_hidden=None, affine_head=False):
        super().__init__()
        self.p = p
        # A buffer, so the head is saved in the state_dict and restored on load.
        if affine_head:
            self.register_buffer("affine_head", torch.tensor(1.0))
        if branch_hidden is None:
            if latent_dim is not None and latent_dim != p:
                raise ValueError(f"without a branch MLP, p must equal the latent size "
                                 f"(p={p}, latent_dim={latent_dim})")
            self.branch = nn.Identity()
        else:
            self.branch = MLP(latent_dim, p, list(branch_hidden))
        self.coord_embed = (RandomFourierEmbedding(coord_dim, dim, n_fourier, fourier_sigma)
                            if rff else None)
        if hidden is None:
            if not rff or p != dim:
                raise ValueError(f"trunk_hidden=None needs --rff and p == dim "
                                 f"(here p={p}, dim={dim}, i.e. n_latent must be 1)")
            self.trunk = nn.Identity()
        else:
            self.trunk = MLP(dim if rff else coord_dim, p, list(hidden))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z_tokens, coords):
        c = self.branch(z_tokens.flatten(1))
        t = self.trunk(self.coord_embed(coords) if self.coord_embed is not None else coords)
        if t.dim() == 2:
            return c @ t.T + self.bias
        return (c.unsqueeze(1) * t).sum(dim=-1) + self.bias


class DotOperator(TBDOperator):
    """`TBDOperator` whose `dec_I` is the affine junction above."""

    def __init__(self, ae, I_scale=1.0, trunk_hidden=(512, 512, 512, 512),
                 n_fourier=64, fourier_sigma=3.0, rff=True, branch_hidden=None,
                 p_basis=None, affine_head=False):
        super().__init__(ae, I_scale=I_scale, dec_depth=1, heads=1, mlp_ratio=1.0,
                         n_fourier=n_fourier, fourier_sigma=fourier_sigma,
                         affine=affine_head)
        latent_dim = ae.n_latent * ae.dim
        self.dec_I = DotDecoder(p_basis or latent_dim, dim=ae.dim, hidden=trunk_hidden,
                                n_fourier=n_fourier, fourier_sigma=fourier_sigma, rff=rff,
                                latent_dim=latent_dim, branch_hidden=branch_hidden,
                                affine_head=affine_head)

    def predict_I(self, z_tokens, coords):
        """softplus(dec_I) * I_scale, or dec_I * I_scale with an affine head."""
        raw = self.dec_I(z_tokens, self._norm_coords(coords))
        if hasattr(self.dec_I, "affine_head"):
            return raw * self.I_scale
        return F.softplus(raw) * self.I_scale

    def init_dec_I_from_recon(self):
        raise NotImplementedError("no attention block to warm-start in the dot decoder")


def _infer_mlp(sd, prefix):
    """(hidden widths, in_dim, out_dim) of the MLP stored under `prefix`,
    or None when no weight carries it (nn.Identity)."""
    idx = sorted({int(k.split(".")[2]) for k in sd
                  if k.startswith(f"{prefix}.network.") and k.endswith(".weight")})
    if not idx:
        return None
    w = [sd[f"{prefix}.network.{i}.weight"] for i in idx]
    return [x.shape[0] for x in w[:-1]], w[0].shape[1], w[-1].shape[0]


def load_dot_operator(path, device="cpu"):
    """(operator, encoder_wrapper); the architecture is inferred from the tensor shapes."""
    state = torch.load(path, map_location=device, weights_only=False)
    arch, gs, sd = state["arch"], state["grid_shape"], state["dec_I"]
    ae = FunctionAutoencoder(grid_shape=gs, in_channels=2, **arch).to(device)
    ae.load_state_dict(state["ae"])

    trunk, branch = _infer_mlp(sd, "trunk"), _infer_mlp(sd, "branch")
    rff = "coord_embed.B" in sd
    n_fourier = sd["coord_embed.B"].shape[1] if rff else arch["n_fourier"]
    p_basis = trunk[2] if trunk else arch["dim"]

    op = DotOperator(ae, I_scale=state["I_scale"],
                     trunk_hidden=(trunk[0] if trunk else None),
                     branch_hidden=(branch[0] if branch else None), p_basis=p_basis,
                     n_fourier=n_fourier, fourier_sigma=arch["fourier_sigma"],
                     rff=rff, affine_head="affine_head" in sd).to(device)
    op.dec_I.load_state_dict(sd)
    op.xy_lo.copy_(state["xy_lo"].to(device))
    op.xy_hi.copy_(state["xy_hi"].to(device))
    op.eval()

    encoder = TBDEncoder(grid_shape=gs, patch=arch["patch"], dim=arch["dim"],
                              n_latent=arch["n_latent"], depth=arch["depth"],
                              dec_depth=arch["dec_depth"], heads=arch["heads"],
                              mlp_ratio=arch["mlp_ratio"], n_fourier=arch["n_fourier"],
                              fourier_sigma=arch["fourier_sigma"], device=device)
    encoder.model = ae
    mean, std = state["stats"]
    encoder.stats = (mean.to(device), std.to(device))
    encoder.fitted = True
    return op, encoder


def run_train(args, device):
    from utils import CHECKPOINTS_DIR
    print("\n--- MODE TRAIN (TBD encoder -> AFFINE branch.trunk junction) ---")
    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    data_train, data_val = splits["data"]
    y_full = splits["y_full"]

    src = args.pretrained_encoder
    enc_path = src if os.path.exists(src) else os.path.join(
        CHECKPOINTS_DIR, f"encoder_tbd_{src}.pt")
    if not os.path.exists(enc_path):
        raise SystemExit(f"pretrained encoder not found: {enc_path}")
    print(f"[operator] Pretrained (reconstruction) encoder: {enc_path}")
    encoder = TBDEncoder.load(enc_path, device=device)

    I_train = data_train["I_target"]
    I_nz = I_train[I_train > 1e-3]
    median_I = torch.median(I_nz).item() if len(I_nz) > 0 else 1.0
    print(f"I_scale (median of I>1e-3) = {median_I:.4e}")

    operator = DotOperator(encoder.model, I_scale=median_I, trunk_hidden=args.trunk_hidden,
                           branch_hidden=args.branch_hidden, p_basis=args.p_basis,
                           n_fourier=encoder.n_fourier, fourier_sigma=encoder.fourier_sigma,
                           rff=args.rff, affine_head=args.affine).to(device)
    operator.set_coord_norm(y_full)
    freeze = args.freeze_encoder
    dec = operator.dec_I
    n_dec = sum(p.numel() for p in dec.parameters())
    trunk = ("NONE, t(y) = the RFF projection itself" if args.trunk_hidden is None
             else f"{list(args.trunk_hidden)}"
                  f"{' on RFF' if args.rff else ' on raw (x,y,phi)'}")
    branch = ("NONE (z IS the coefficient vector)" if args.branch_hidden is None
              else f"{list(args.branch_hidden)}")
    print(f"Operator: encoder {'FROZEN' if freeze else 'FINE-TUNED'}, "
          f"m = n_latent {encoder.n_latent} x dim {encoder.dim} = {encoder.latent_dim}, "
          f"p = {dec.p}\n"
          f"  branch (A) {branch} -> {sum(x.numel() for x in dec.branch.parameters()):,} params\n"
          f"  trunk      {trunk} -> {sum(x.numel() for x in dec.trunk.parameters()):,} params\n"
          f"  head       {'LINEAR (no softplus) -> affine reconstructor' if args.affine else 'softplus'}\n"
          f"  {n_dec:,} decoder params total, data_loss={args.data_loss}, query=(x, y, phi)")

    path = operator_path(args.name)
    fit_operator(
        operator, encoder,
        {"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]},
        {"u1_in": data_val["u1_in"], "u2_in": data_val["u2_in"]},
        y_full, I_train, data_val["I_target"],
        epochs=args.op_epochs, batch_size=args.op_batch, lr=args.op_lr,
        n_query=args.op_n_query, freeze_encoder=freeze, device=device,
        val_every=args.op_val_every, encoder_lr=args.op_encoder_lr, save_path=path,
        w_phys=0.0, w_bc=0.0, w_data=1.0, monitor_phys=args.monitor_phys,
        history_name=args.name, resume=args.resume, data_loss=args.data_loss)

    save_operator(operator, encoder, path,
                  meta={"pretrained": src, "freeze": freeze, "decoder": "dot",
                        "trunk_hidden": args.trunk_hidden, "rff": bool(args.rff),
                        "branch_hidden": args.branch_hidden, "p_basis": args.p_basis,
                        "affine": bool(args.affine),
                        "data_loss": args.data_loss, "query": "xy_phi"})
    print(f"[OK] Evaluate with: python {os.path.basename(__file__)} --mode eval "
          f"--name {args.name}")


def relative_errors(operator, encoder, splits, which, device, batch_size=128):
    from torch.utils.data import DataLoader, TensorDataset
    data = split_data(splits, which)
    ev = Evaluator(operator, device, h5_path=solver_h5())
    ev.encoder = encoder
    enc = encoder.transform_dicts({"u1_in": data["u1_in"], "u2_in": data["u2_in"]}, INPUT_KEYS)
    loader = DataLoader(TensorDataset(enc["u1_in"], enc["u2_in"], data["I_target"]),
                        batch_size=batch_size, shuffle=False)
    return ev.evaluate_split_errors(loader, y_coords=splits["y_full"], label=which.upper())


def sqerr_cached(operator, encoder, splits, which, device, name, refresh=False):
    import model_errors
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"sqerr_{name}_{which}.npy")
    M = split_data(splits, which)["I_target"].shape[0]
    if not refresh and os.path.exists(path):
        a = np.load(path)
        if len(a) == M:
            return a
    se = model_errors.squared_errors(operator, encoder, splits, device, which, batch_size=32)
    np.save(path, se)
    return se


def reference_sqerr(name, which):
    """Cached ||I - N(u)||^2 of a model from `lower_bound/cache`, or None (read-only)."""
    for base in (CACHE, os.path.join(_LB, "cache")):
        npy = os.path.join(base, f"sqerr_{name}_{which}.npy")
        if os.path.exists(npy):
            return np.load(npy)
        npz = os.path.join(base, f"metrics_{name}_{which}.npz")
        if os.path.exists(npz):
            return np.load(npz)["sqerr"]
    return None


def reference_meta(name):
    """(kind, p, affine) of a model of the main checkpoints/, or None."""
    from encoders import _ENCODER_EXT, encoder_run
    from deeponet import infer_architecture
    from utils import CHECKPOINTS_DIR

    if os.path.exists(operator_path(name)):
        arch = torch.load(operator_path(name), map_location="cpu",
                          weights_only=False)["arch"]
        return "dot", int(arch["n_latent"]) * int(arch["dim"]), True
    op = os.path.join(CHECKPOINTS_DIR, f"operator_tbd_{name}.pt")
    if os.path.exists(op):
        arch = torch.load(op, map_location="cpu", weights_only=False)["arch"]
        return "tbd", int(arch["n_latent"]) * int(arch["dim"]), False
    don = os.path.join(CHECKPOINTS_DIR, f"{name}.pth")
    if not os.path.exists(don):
        return None
    kind = next((k for k, ext in _ENCODER_EXT.items()
                 if os.path.exists(os.path.join(
                     CHECKPOINTS_DIR, f"encoder_{k}_{encoder_run(name)}.{ext}"))),
                None)
    if kind is None:
        return None
    return kind, int(infer_architecture(torch.load(don, map_location="cpu"))[2]), True


def spectrum(splits, which):
    """lambda_k of Gamma_hat for the split, from `lower_bound/cache` when available."""
    import bound
    path = os.path.join(_LB, "cache", f"lam_{which}.npy")
    I_S = split_data(splits, which)["I_target"]
    if os.path.exists(path):
        lam = np.load(path)
        if len(lam) == I_S.shape[0]:
            return {"lam": lam, "M": len(lam), "d": None}
    return bound.eigenvalues(I_S, tag=which, verbose=True)


def run_eval(args, device):
    import bound
    import model_errors
    import run as lb_run

    operator, encoder = load_dot_operator(operator_path(args.name), device=device)
    p = encoder.latent_dim
    print(f"Operator loaded: {operator_path(args.name)} | p = {p} | "
          f"I_scale = {float(operator.I_scale):.4e}")
    splits = load_and_split_dataset(args.dataset, map_location="cpu")

    print(f"\n{'='*72}\n RELATIVE ERRORS (per snapshot)\n{'='*72}")
    rel = {}
    for which in args.eval_splits:
        rel[which] = relative_errors(operator, encoder, splits, which, device,
                                     batch_size=args.batch_size_data)

    print(f"\n{'='*72}\n THEOREM 3.4 -- E_hat AGAINST THE BOUND\n{'='*72}")
    spec = {t: spectrum(splits, t) for t in SETS}
    curves = {t: bound.energy_curve(spec[t]) for t in SETS}
    mine = {t: model_errors.E_hat(sqerr_cached(operator, encoder, splits, t, device,
                                               args.name, args.refresh))
            for t in SETS}

    rows = [{"name": args.name, "kind": "dot", "p": p, "affine": True,
             "ehat": mine, "bound": {t: bound.lower_bound(spec[t], p) for t in SETS},
             "p_star": {t: lb_run.p_star(curves[t], mine[t]) for t in SETS}}]
    for name in REFERENCES:
        if name == args.name:
            continue
        meta = reference_meta(name)
        e = {t: reference_sqerr(name, t) for t in SETS}
        if meta is None or any(v is None for v in e.values()):
            print(f"  [reference skipped: {name}]")
            continue
        kind, q, affine = meta
        e = {t: model_errors.E_hat(v) for t, v in e.items()}
        rows.append({"name": name, "kind": kind, "p": q, "affine": affine, "ehat": e,
                     "bound": {t: bound.lower_bound(spec[t], q) for t in SETS},
                     "p_star": {t: lb_run.p_star(curves[t], e[t]) for t in SETS}})

    print("\nE^2 = (1/M) sum_m ||I_m - N(u_m)||^2  >=  sum_(k>p) lambda_k   "
          "if the reconstructor is AFFINE (Thm 3.4)\n")
    print(lb_run.table_bound(rows))
    print(f"\n† nonlinear reconstructor (attention decoder): the bound does NOT constrain it.\n"
          f"  {args.name} has an AFFINE junction -> E/bound >= 1 is expected.")

    if args.out:
        lb_run.FAMILY["dot"] = ("#b23a48", "P", 9.0, "DeepONet junction (TBD encoder)")
        lb_run.KINDS = ("pca", "autoencoder", "tbd", "dot")
        out = lb_run.plot_bound_comparison(spec, curves, rows, args.out)
        print(f"Figure: {out}")

    print(f"\n{'='*72}\n SUMMARY\n{'='*72}")
    for which in args.eval_splits:
        v = rel[which]
        print(f"  {which.upper():<6} I-L2 = {v['I_l2']*100:6.2f} %    "
              f"G-L2 = {v['G_l2']*100:6.2f} %")
    for t in SETS:
        print(f"  {SETS[t]:<6} E_hat = {mine[t]:.4e}   bound(p={p}) = "
              f"{bound.lower_bound(spec[t], p):.4e}   ratio = "
              f"{mine[t] / bound.lower_bound(spec[t], p):.2f}")


def build_parser():
    p = argparse.ArgumentParser(description="Perceiver encoder + branch.trunk junction.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["train", "eval"])
    p.add_argument("--name", default="abl-dot")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--device", default=None)

    p.add_argument("--pretrained_encoder", default="TBD-N1-D128",
                   help="encoder NAME (checkpoints/encoder_tbd_<name>.pt) or a path")
    p.add_argument("--trunk_hidden", type=int, nargs="*", default=[512, 512, 512, 512],
                   help="hidden widths of the trunk MLP. Passed EMPTY (--trunk_hidden), "
                        "there is no trunk at all: t(y) is the RFF projection, dotted "
                        "straight with z (needs n_latent = 1)")
    p.add_argument("--branch_hidden", type=int, nargs="*", default=None,
                   help="hidden widths of the branch MLP A: R^(n_latent*dim) -> R^p. "
                        "Absent (the default), there is no approximator and z is used "
                        "straight as the coefficient vector")
    p.add_argument("--affine", action="store_true",
                   help="drop the softplus: I = (<beta, gamma(y)> + b) * I_scale, a TRUE "
                        "affine reconstructor covered by Thm 3.4")
    p.add_argument("--p_basis", type=int, default=None,
                   help="width of the affine junction (default: n_latent*dim). Only "
                        "meaningful together with --branch_hidden")
    p.add_argument("--rff", action=argparse.BooleanOptionalAction, default=True,
                   help="feed the trunk with the random Fourier embedding (default) "
                        "or with the raw (x, y, phi)")
    p.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--op_epochs", type=int, default=2000)
    p.add_argument("--op_lr", type=float, default=1e-3)
    p.add_argument("--op_encoder_lr", type=float, default=1e-4)
    p.add_argument("--op_batch", type=int, default=64)
    p.add_argument("--op_n_query", type=int, default=8192)
    p.add_argument("--op_val_every", type=int, default=25)
    p.add_argument("--data_loss", default="rel_l2", choices=["rel_l2", "mse"])
    p.add_argument("--monitor_phys", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--eval_splits", nargs="+", default=["test"],
                   choices=["train", "val", "test"])
    p.add_argument("--batch_size_data", type=int, default=128)
    p.add_argument("--refresh", action="store_true", help="recompute the cached E_hat")
    p.add_argument("--out", default=os.path.join(_HERE, "bound_comparison_dot.pdf"),
                   help="output figure ('' to skip it)")
    return p


def main():
    args = build_parser().parse_args()
    if not args.trunk_hidden:
        args.trunk_hidden = None
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DEVICE: {str(device).upper()} ===")
    os.makedirs(CKPT, exist_ok=True)
    if args.mode == "train":
        run_train(args, device)
    else:
        run_eval(args, device)


if __name__ == "__main__":
    main()
