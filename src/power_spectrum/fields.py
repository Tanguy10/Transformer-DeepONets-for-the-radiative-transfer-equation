"""Supplies I as (B, n_dirs, 150, 150) images, from the solver or from a model.

Shaping only. Entry points: grid_geometry, angles_deg, load_source, predict_I,
truth_I, to_images, to_G.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

GRID = 150

# Every source must be a model trained against the (x, y, phi) dataset: a 4-column
# checkpoint from the historical code base cannot be queried with it.
SOURCES = {
    "truth": {"label": "truth (FVDOM)", "kind": "truth"},
    # PCA and AE at the same p and the same 500-component branch input, so the
    # only difference left between the two panels is the encoder.
    "pca": {"label": "PCA-DeepONet", "kind": "deeponet",
            "cfg": {"name": "PCA-p512-softplus", "encoder": "pca", "arch": "modular"}},
    "ae": {"label": "AE-DeepONet", "kind": "deeponet",
           "cfg": {"name": "AE-p512-softplus", "encoder": "autoencoder", "arch": "modular"}},
    "tbd": {"label": "Transformer-Based DeepONet", "kind": "operator",
            "ckpt": "operator_tbd_TBD-N4-D128.pt"},
}
MODEL_KEYS = ("pca", "ae", "tbd")

# Families of train.py, as they may prefix a run name on the command line.
FAMILY = {"pca": "pca", "ae": "autoencoder", "tbd": "tbd"}


def detect_family(run):
    """Family of a run name, read off the checkpoints on disk."""
    import utils
    from deeponet_joint import operator_path
    from encoders import encoder_path

    found = ["tbd"] if os.path.exists(operator_path(run)) else []
    if os.path.exists(os.path.join(utils.CHECKPOINTS_DIR, f"{run}.pth")):
        found += [f for f in ("pca", "ae")
                  if os.path.exists(encoder_path(run, FAMILY[f]))]
    if not found:
        raise SystemExit(
            f"No checkpoint for run {run!r} in {utils.CHECKPOINTS_DIR}.\n"
            f"Train it first, e.g.: python src/train.py --model tbd --name {run}")
    if len(found) > 1:
        raise SystemExit(f"Run {run!r} matches several families "
                         f"({', '.join(found)}); prefix it, e.g. {found[0]}:{run}")
    return found[0]


def resolve_source(spec):
    """SOURCES key of a --sources entry, registering the run when it is a new one.

    A key of the table above is returned as is; anything else is a run name given
    to train.py, optionally prefixed by its family ('tbd:my-run', needed only when
    one name carries several), and becomes a source named after the run."""
    import utils
    from deeponet_joint import operator_path

    if spec in SOURCES:
        return spec
    family, run = spec.split(":", 1) if ":" in spec else (detect_family(spec), spec)
    if family not in FAMILY:
        raise SystemExit(f"Unknown family {family!r} in {spec!r} "
                         f"({' | '.join(FAMILY)}).")
    if run in SOURCES:
        return run
    path = (operator_path(run) if family == "tbd"
            else os.path.join(utils.CHECKPOINTS_DIR, f"{run}.pth"))
    if not os.path.exists(path):
        raise SystemExit(f"Missing checkpoint: {path}\nTrain it first, e.g.: "
                         f"python src/train.py --model {family} --name {run}")
    if family == "tbd":
        SOURCES[run] = {"label": run, "kind": "operator",
                        "ckpt": os.path.basename(path)}
    else:
        SOURCES[run] = {"label": run, "kind": "deeponet",
                        "cfg": {"name": run, "encoder": FAMILY[family],
                                "arch": "modular"}}
    return run

# Memory per forward scales as batch * chunk, not chunk alone; 300k fits in
# ~1.5 GB so the analysis still runs while a training occupies the GPU.
PTS_PER_FWD = 300_000


def grid_geometry(y_full, grid=GRID):
    """Geometry from the output coordinates; raises if the mesh is not uniform
    Cartesian and cell-major, which the ring FFT requires."""
    y = torch.as_tensor(y_full).cpu().numpy()
    n_dirs = int(np.unique(y[:, 2]).shape[0])
    if y.shape[0] % n_dirs:
        raise ValueError(f"{y.shape[0]} points not divisible by {n_dirs} directions")

    phis = y[:n_dirs, 2]
    if not np.allclose(y[n_dirs:2 * n_dirs, 2], phis):
        raise ValueError("y is not cell-major")

    cells = y[::n_dirs, :2]
    if cells.shape[0] != grid * grid:
        raise ValueError(f"{cells.shape[0]} cells, expected {grid}^2")

    xs, ys = np.unique(cells[:, 0]), np.unique(cells[:, 1])
    dx, dy = np.diff(xs), np.diff(ys)
    if not (np.allclose(dx, dx[0], rtol=1e-3) and np.allclose(dy, dy[0], rtol=1e-3)):
        raise ValueError("non-uniform mesh: the grid FFT does not apply")

    X = cells[:, 0].reshape(grid, grid)
    Y = cells[:, 1].reshape(grid, grid)
    if not (np.ptp(X[0]) > 0 and np.ptp(Y[0]) == 0):
        raise ValueError("reshape(150,150) does not yield [iy, ix]")

    dx0, dy0 = float(dx.mean()), float(dy.mean())
    return {"grid": grid, "n_dirs": n_dirs, "dx": dx0, "dy": dy0,
            "Lx": grid * dx0, "Ly": grid * dy0}


def angles_deg(y_full, n_dirs):
    """phi as stored, in degrees, and the permutation that sorts them: the dataset
    order is not monotonic in phi, and unsorted panels hide the rotation.

    phi is read straight from column 2 (convention utils.dirs_from_phi: angle from
    +y, clockwise). The pre-phi revision of this file labelled panels with
    atan2(muy, mux), i.e. the angle from +x -- the two differ by phi = 90 - alpha,
    so panel names shift accordingly."""
    p = np.asarray(torch.as_tensor(y_full).cpu())[:n_dirs, 2]
    phi = np.degrees(p) % 360.0
    order = np.argsort(phi)
    return phi[order], order


def to_images(i_flat, geom):
    """(B, n_cells*n_dirs) -> (B, n_dirs, grid, grid), float64."""
    g, nd = geom["grid"], geom["n_dirs"]
    a = np.asarray(i_flat, dtype=np.float64).reshape(-1, g * g, nd)
    return np.ascontiguousarray(a.transpose(0, 2, 1)).reshape(-1, nd, g, g)


def to_G(images, geom):
    """(B, n_dirs, grid, grid) -> (B, grid, grid): G = sum_m I_m * w.

    Uniform quadrature on the ordinates with the solver's solid-angle weight
    w = 4*pi/n_dirs (utils.dir_weight), so G is the fvDOM `G` field itself.
    """
    return np.asarray(images).sum(axis=1) * (4.0 * np.pi / geom["n_dirs"])


def load_source(key, device):
    """(model, encoder) for a SOURCES key, or (None, None) for the truth."""
    import utils
    src = SOURCES[key]
    if src["kind"] == "truth":
        return None, None
    ckpt_dir = src.get("dir", utils.CHECKPOINTS_DIR)
    if src["kind"] == "operator":
        path = os.path.join(ckpt_dir, src["ckpt"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing checkpoint: {path}")
        from tbdeeponet import load_operator
        return load_operator(path, device=device)
    cfg = src["cfg"]
    path = os.path.join(ckpt_dir, f"{cfg['name']}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing checkpoint: {path}")
    if "encoder_path" in src:
        from deeponet import load_model
        from encoders import load_encoder
        return (load_model(path, device),
                load_encoder(cfg["encoder"], src["encoder_path"], device=device))
    return utils.load_any_model(cfg, device), utils.load_any_encoder(cfg, device)


def predict_I(model, encoder, dict_split, sl, y_full, device,
              pts_per_fwd=PTS_PER_FWD):
    """Predicted flat I (B, 360000) for a slice of snapshots, numpy float64."""
    from deeponet import INPUT_KEYS

    enc = encoder.transform_dicts(
        {"u1_in": dict_split["u1_in"][sl], "u2_in": dict_split["u2_in"][sl]},
        INPUT_KEYS)
    u_in = (enc["u1_in"].to(device), enc["u2_in"].to(device))

    b = u_in[0].shape[0]
    chunk = max(2000, int(pts_per_fwd) // max(b, 1))
    y = torch.as_tensor(y_full)
    out = []
    with torch.no_grad():
        for k0 in range(0, y.shape[0], chunk):
            p = model(u_in, y[k0:k0 + chunk].to(device))
            out.append((p.unsqueeze(0) if p.dim() == 1 else p).float().cpu())
    return torch.cat(out, dim=1).numpy().astype(np.float64)


def truth_I(dict_split, sl):
    """Solver reference, flat (B, 360000)."""
    return dict_split["I_target"][sl].to(torch.float64).numpy()


def rel_l2(pred_flat, true_flat):
    """Relative L2 error per snapshot."""
    d = np.linalg.norm(pred_flat - true_flat, axis=1)
    return d / np.maximum(np.linalg.norm(true_flat, axis=1), 1e-12)
