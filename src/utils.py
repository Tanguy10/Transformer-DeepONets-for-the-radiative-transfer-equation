import os
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import QuadMesh
from matplotlib.contour import ContourSet
from matplotlib.image import AxesImage
import h5py
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_DATASET = os.path.join(DATA_DIR, "dataset_train_full_phi.pt")
MESH_STUB = os.path.join(DATA_DIR, "mesh_stub.h5")
SOLVER_G_PT = os.environ.get("RTE2D_SOLVER_G", os.path.join(DATA_DIR, "G_solver.pt"))
CHECKPOINTS_DIR = os.path.join(HERE, "checkpoints")
RESULTS_DIR = os.path.join(HERE, "results")

N_DIRS_DATA = 16


QUERY_DIM = 3


MAP_TITLE_SIZE = 16


FIG_FORMAT = os.environ.get("RTE2D_FIG_FORMAT", "pdf").lower().lstrip(".")
FIG_DPI = int(os.environ.get("RTE2D_FIG_DPI", "600"))
FIG_RASTERIZE = os.environ.get("RTE2D_FIG_RASTER", "1") != "0"
ERR_VMAX_DEFAULT = 20.0
ERR_VMAX = float(os.environ.get("RTE2D_ERR_VMAX", "0")) or None
_HEAVY_ARTISTS = (ContourSet, QuadMesh, AxesImage)


def save_figure(path, fig=None, **kw):
    """The single figure writer of the repository: it, and not the call site, owns
    the output format. The extension of `path` is replaced by FIG_FORMAT (pdf by
    default: axes, text and curves stay VECTOR, hence sharp at any zoom, which no
    finite dpi gives), and the field maps (contourf/imshow/pcolormesh) are rasterized
    at FIG_DPI inside that vector page -- a 50-level contourf of a 150x150 grid
    weighs several MB in pure vector, ~5x less rasterized at 600 dpi, and the data
    itself only has 150 points per side.

    Both knobs are environment variables, so a PNG is one run away without touching
    any code:  RTE2D_FIG_FORMAT=png RTE2D_FIG_DPI=1200 python evaluate.py ...
    (RTE2D_FIG_RASTER=0 forces everything vector).

    Returns the path actually written -- rebind the caller's variable, so what gets
    printed is what exists on disk."""
    fig = plt.gcf() if fig is None else fig
    out = f"{os.path.splitext(path)[0]}.{FIG_FORMAT}"
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    if FIG_RASTERIZE and FIG_FORMAT != "png":
        with warnings.catch_warnings():
            # ContourSet.draw carries no rasterization decorator, matplotlib warns
            # that it will be ignored -- it is not, the map does end up as an image.
            warnings.filterwarnings("ignore", message="Rasterization of")
            for ax in fig.get_axes():
                for art in ax.get_children():
                    if isinstance(art, _HEAVY_ARTISTS):
                        art.set_rasterized(True)
    fig.savefig(out, dpi=FIG_DPI, **kw)
    return out


def dirs_from_phi(phi):
    """THE angular convention of this repository: the only angular degree of freedom
    of the 2D RTE is phi, the angle of the PROPAGATION direction measured from +y
    CLOCKWISE (phi = atan2(mu_x, mu_y) folded into [0, 2pi)). It is the angle of the
    FVDOM ordinates of the H5 file, phi_m = 2*pi*m/16 (cf. `ordinate_phis`).

    Query points are ALWAYS stored as (x, y, phi) -- 3 columns: that is what the
    dataset holds, what `build_collocation` produces, and what the trunk (or the
    coordinate decoder) receives. Unlike the (mu_x, mu_y) unit-circle embedding, phi
    does NOT make I structurally 2pi-periodic: the network has to learn the
    periodicity, and it cannot learn it at the phi = 0 = 2pi cut, which no data
    straddles (it separates m=15 from m=0, which are NEIGHBOURS on the circle).

    This function is the ONLY place that rebuilds the direction vector
    (mu_x, mu_y) = (sin phi, cos phi) that mu.grad(I) needs: the physics wants the
    vector, the parameterisation stays phi. Never hand-write a direction sine/cosine."""
    p = np.asarray(phi, dtype=np.float64)
    return np.stack([np.sin(p), np.cos(p)], axis=-1)


def transport_coeffs(phi):
    """(sin phi, cos phi) in torch, for mu.grad(I) = sin phi dI/dx + cos phi dI/dy.
    The torch counterpart of `dirs_from_phi`: differentiable, same conventions."""
    return torch.sin(phi), torch.cos(phi)


def ordinate_phis(n_dirs=N_DIRS_DATA):
    """phi of the FVDOM ordinates of the H5, in the order of the m index of
    I[:, :, m]: m turns CLOCKWISE from +y, so phi_m = 2*pi*m/n_dirs.

    n_dirs=16 gives back the 16 training directions; any n_dirs that is a multiple of
    4 contains all of them."""
    return 2 * math.pi * np.arange(n_dirs) / n_dirs


def dir_weight(n_dirs=N_DIRS_DATA):
    """Quadrature weight of one ordinate for the incident radiation G.

    fvDOM's G is the SOLID-ANGLE integral G = int_{4pi} I dOmega, which the solver
    evaluates as sum_m I_m * 4*pi/n_dirs. Checked against the `G` field of the solver
    H5 over 80 snapshots: the identity holds to 0.87 eps(float32), i.e. exactly.

    In the equatorial plane I does not depend on theta, so this is twice the azimuthal
    integral int_0^{2pi} I dphi. Relative errors on G are insensitive to the factor,
    absolute ones are not."""
    return 4.0 * math.pi / n_dirs


def load_and_split_dataset(dataset_path=DEFAULT_DATASET,
                           train_frac=0.70, val_frac=0.15, seed=42, map_location="cpu"):
    """
    Load the whole dataset (dict_pde / dict_bc / dict_data) and split it into
    train/val/test with one GLOBAL, reproducible shuffle (the same indices for the
    three dictionaries).

    Default split: 70 % train / 15 % val / 15 % test (test = the remainder). The three
    blocks are CONSECUTIVE slices of the same `randperm(n, seed=42)` permutation as
    the historical 85/15 split, in the order train -> val -> test. Intended
    consequence:

        train_85 = train_70 + val_15        and        test_15 = val_85

    in other words the test block is exactly the old validation set, so checkpoints
    trained under the 85/15 split never saw it and stay honestly evaluable on it
    without retraining.

    Returns a dict:
        {
          "pde":  (dict_pde_train,  dict_pde_val),
          "bc":   (dict_bc_train,   dict_bc_val),
          "data": (dict_data_train, dict_data_val),
          "pde_test" / "bc_test" / "data_test": the dicts of the test block,
          "y_full": y_data,            # shared grid coordinates
          "train_idx" / "val_idx" / "test_idx": indices into the source dataset,
        }

    The test block is exposed separately (rather than as a 3rd tuple element) so that
    existing callers writing `data_train, data_val = splits["data"]` keep working
    unchanged.

    The "y" key (the shared grid) is never indexed by snapshot indices.

    Peak memory is kept in check (several trainings may share the machine):
      - mmap=True: the tensors of the .pt are mapped from disk instead of being copied
        into anonymous RAM; only the pages actually read go through the page cache,
        which the kernel can evict. The original dataset no longer counts towards the
        peak.
      - incremental split: each source tensor is dropped from the source dict as soon
        as its train/val/test copies exist, instead of coexisting with ALL the copies
        until the end (~2x the dataset size otherwise). The three copies partition the
        source tensor, so their combined cost stays that of the dataset.
    """
    try:
        dataset = torch.load(dataset_path, map_location=map_location, mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        dataset = torch.load(dataset_path, map_location=map_location, weights_only=False)
    dict_pde = dataset["dict_pde"]
    dict_bc = dataset["dict_bc"]
    dict_data = dataset["dict_data"]

    n_total = dict_data["u1_in"].shape[0]
    torch.manual_seed(seed)
    indices = torch.randperm(n_total)

    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    y_full = dict_data["y"]

    def _split_three(d):
        d_train, d_val, d_test = {}, {}, {}
        for k in list(d.keys()):
            v = d.pop(k)
            if k == "y":
                d_train[k] = d_val[k] = d_test[k] = v
            else:
                d_train[k] = v[train_idx]
                d_val[k] = v[val_idx]
                d_test[k] = v[test_idx]
            del v
        return d_train, d_val, d_test

    pde_train, pde_val, pde_test = _split_three(dict_pde)
    bc_train, bc_val, bc_test = _split_three(dict_bc)
    data_train, data_val, data_test = _split_three(dict_data)

    return {
        "pde": (pde_train, pde_val),
        "bc": (bc_train, bc_val),
        "data": (data_train, data_val),
        "pde_test": pde_test,
        "bc_test": bc_test,
        "data_test": data_test,
        "y_full": y_full,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


SOLVER_H5 = os.environ.get("RTE2D_SOLVER_H5", "")
# Mountpoint SERVING that path when the H5 lives on a remote filesystem. Both are
# site-specific, hence empty by default: unset, the repository runs on `mesh_stub.h5`
# and never touches the raw solver output. See `solver_h5`.
SOLVER_H5_MOUNT = os.environ.get("RTE2D_SOLVER_H5_MOUNT", "")


def _mounted(mountpoint):
    """True if `mountpoint` appears in /proc/mounts. This reads a procfs file, so it
    touches no remote filesystem and cannot block."""
    try:
        with open("/proc/mounts") as f:
            return any(ln.split()[1] == mountpoint for ln in f if len(ln.split()) > 1)
    except OSError:
        return False


def solver_h5(path=SOLVER_H5, mount=SOLVER_H5_MOUNT, fallback=MESH_STUB, probe=None):
    """The solver H5 if probing is allowed AND it answers, otherwise `mesh_stub.h5`.
    By default **it is not probed**.

    WARNING never replace this guard with `os.path.exists(path)`. That H5 typically
    lives on a remote filesystem served through an indirect autofs. When the host does
    not answer -- which can last for months -- the SLIGHTEST syscall on it
    (`exists`, `stat`, even `lstat` on the link: it is the PARENT directory that is
    mounted) goes into UNINTERRUPTIBLE disk sleep (state D, wchan autofs_wait). The
    process then stops answering SIGKILL and only comes back when autofs feels like it.

    Checking /proc/mounts is NOT enough: an NFS mount that is *present* but whose
    server is dead blocks in exactly the same way. Hence the inverted default -- we
    assume the H5 is absent, and only probe it on explicit request:

        RTE2D_PROBE_SOLVER_H5=1    (environment variable)   or   solver_h5(probe=True)

    This has no effect on the numbers: errors are computed on `y_coords`. The H5 is
    only used to plot on the solver grid, and `Evaluator.ensure_grid` rebuilds that
    grid from `y_full` when we are on the stub."""
    if probe is None:
        probe = os.environ.get("RTE2D_PROBE_SOLVER_H5") == "1"
    if not probe or not path or not _mounted(mount):
        return fallback
    return path if os.path.exists(path) else fallback


def solver_G(indices, h5_path=None):
    """The `G` field WRITTEN BY THE SOLVER, for the given snapshot indices: (n, n_cells).

    Snapshot i of the .pt is the i-th `time_*` group in ALPHABETICAL order -- the order
    `prepare_data` stacks them in -- so `indices` are indices into the .pt.

    G is the solid-angle integral of the SAME 16 intensities: checked over the 901
    snapshots of the test split, `G == dir_weight(16) * sum_m I_m` to 0.88 eps(float32).
    Reading it therefore changes no number; it only makes the reference literal.

    Read from the LOCAL copy `data/G_solver.pt` when it is there (`import_G`-style dump:
    {"G": (n_total, n_cells), "groups": the sorted `time_*` names}); that copy is a bitwise
    identical extract of the H5, so nothing changes and the remote store stays out of the loop.
    Falls back to the H5 itself, which needs RTE2D_SOLVER_H5 -- it sits behind an autofs
    mount that can hang the process beyond SIGKILL (see `solver_h5`)."""
    idx = np.asarray(indices).ravel()
    if h5_path is None and os.path.exists(SOLVER_G_PT):
        return torch.load(SOLVER_G_PT, weights_only=False)["G"][idx].float()
    path = h5_path or SOLVER_H5
    if not path:
        raise RuntimeError(
            f"g_ref='solver' needs either {SOLVER_G_PT} or the raw solver H5 "
            "(export RTE2D_SOLVER_H5=/path/to.h5)")
    with h5py.File(path, "r") as f:
        groups = sorted(k for k in f if k.startswith("time_"))
        G = np.stack([f[groups[i]]["G"][:] for i in idx])
    return torch.from_numpy(G).float()


def split_data(splits, which="test"):
    """The snapshot dict of one split of `load_and_split_dataset`.

    DEFAULT = **test**, and that is the rule of this repository: every REPORTED error
    (I-L2, G-L2, maps, residual) is measured on snapshots that neither training nor
    model selection ever saw. The val split stays reserved for monitoring during
    training and for picking the best checkpoint -- quoting it as performance would
    amount to publishing the selection criterion. The one deliberate exception is
    `lower_bound/`, which compares E_hat (a NON-relative error) with the Thm 3.4 bound
    on train AND test, each with its own spectrum."""
    try:
        return {"train": splits["data"][0],
                "val": splits["data"][1],
                "test": splits["data_test"]}[which]
    except KeyError:
        raise ValueError(f"split inconnu : {which!r} (train | val | test)") from None


ERR_CMAP = "bwr"
ABS_ERR_CMAP = os.environ.get("RTE2D_ABS_ERR_CMAP", "turbo")


def sym_error_levels(arrays, n_levels=21, vmax=None):
    """Contour levels SYMMETRIC around 0, shared by several signed error maps (so
    white lands exactly on 0 and the panels are comparable).

    Returns (levels, vmax) with vmax = max |error| over all `arrays`, unless `vmax`
    (or RTE2D_ERR_VMAX, in %) forces it: one scale imposed on every run makes the
    error maps of several models comparable by eye instead of each rescaling itself.
    """
    arrays = [np.asarray(a) for a in arrays if a is not None]
    measured = max((float(np.max(np.abs(a))) for a in arrays), default=0.0) or 1e-12
    vmax = float(vmax or ERR_VMAX or measured)
    return np.linspace(-vmax, vmax, n_levels), vmax


def map_colorbar(fig, im, ax, label=None, size="4%", pad=0.10, **kw):
    """Colour bar in an axes of ITS OWN, appended to `ax`: it takes exactly the height
    of the map and steals the same width from every panel it is applied to, so the maps
    of a figure all keep the same size whatever their bar carries.
    """
    cax = make_axes_locatable(ax).append_axes("right", size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax, **kw)
    if label is not None:
        cb.set_label(label)
    return cb


def g_simple_figure(G_ref, G_pred, err, X_grid, Y_grid, g_vmax=None):
    """Reference | prediction | |dG| / ||G||_inf, the palette of the multi-resolution
    figures (`robustness_resolution`): inferno for G, ABS_ERR_CMAP for the error, imshow
    rather than filled contours.

    NOTHING here is scaled on the panel it draws, otherwise two figures could not be
    compared by eye: an error of 20 % must carry the same colour everywhere.

      * the error panel runs on a FIXED 0..ERR_VMAX_DEFAULT % (RTE2D_ERR_VMAX
        overrides), whatever the model -- a pointed colour bar flags the models that
        run past it instead of silently rescaling the map;
      * both G panels share `g_vmax`, passed by `plot_eval_snapshots` as the peak over
        ALL the snapshots of the figure set, so one G level = one colour across the
        whole set (it falls back to the peak of the reference, which is at least
        identical from one model to the next on a given snapshot).
    """
    vG = float(g_vmax or np.max(G_ref))
    vE = float(ERR_VMAX or ERR_VMAX_DEFAULT)
    kw = dict(origin="lower", interpolation="nearest",
              extent=[float(X_grid.min()), float(X_grid.max()),
                      float(Y_grid.min()), float(Y_grid.max())])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (field, title) in enumerate(((G_ref, "Reference"), (G_pred, "Prediction"))):
        im = ax[k].imshow(field.reshape(150, 150), cmap="inferno", vmin=0, vmax=vG, **kw)
        ax[k].set_title(title, fontsize=MAP_TITLE_SIZE)
        map_colorbar(fig, im, ax[k], label="G",
                     extend="max" if float(np.max(field)) > vG else "neither")
    im = ax[2].imshow(err.reshape(150, 150), cmap=ABS_ERR_CMAP, vmin=0, vmax=vE, **kw)
    ax[2].set_title("Relative error", fontsize=MAP_TITLE_SIZE)
    map_colorbar(fig, im, ax[2], label=r"|dG| / ||G||$_\infty$  [%]",
                 extend="max" if float(np.max(err)) > vE else "neither")
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


def _err_title(e, label):
    """Title of a signed error map: explicit signed min/max."""
    return f"{label}\n(min {e.min():+.1f}%, max {e.max():+.1f}%)"


def rel_error_map(pred, ref, mode="peak", floor_frac=0.02, signed=True):
    """RELATIVE error map, in PERCENT, between a predicted field and its reference.

    mode:
      * "peak"  (default) -- 100*|pred - ref| / max(ref). A GLOBAL normalisation by the
        peak of the reference field: robust (no division by ~0 in the cold regions,
        where G tends to 0 and a local relative error would blow up without saying
        anything useful), and identical for every model on a given snapshot, hence
        directly comparable.
      * "local" -- 100*|pred - ref| / max(|ref|, floor_frac*max|ref|): the POINTWISE
        relative error, the literal definition, but floored at `floor_frac` of the peak
        so that near-zero cells do not saturate the map.

    signed=True (default) returns the SIGNED error (pred - ref, in %), to be drawn with
    `ERR_CMAP` + `sym_error_levels`: blue = the model under-estimates, red = it
    over-estimates. signed=False returns |pred - ref|.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    peak = float(np.max(np.abs(ref))) or 1.0
    denom = peak if mode == "peak" else np.maximum(np.abs(ref), floor_frac * peak)
    diff = pred - ref
    return 100.0 * (diff if signed else np.abs(diff)) / denom


def snapshot_variance(dict_data, field="G", n_dirs=None, metric="var"):
    """SPATIAL variance of each snapshot of a split dict (a measure of "chaos").

    field: "G" (incident radiation, the integral of I over phi -- the default, and the
    field actually plotted), "I" (raw directional intensities), "T" (u1_in) or "kappa"
    (u2_in).
    metric: "var" (plain variance, what a "largest variance" ranking asks for), "cv"
    (coefficient of variation std/mean) or "structure" (cf. below).

    "var" and "cv" both rank by INTENSITY: they put the compact, very bright plumes
    first, and those all look alike. "structure" divides the field by its own peak
    BEFORE measuring it, so the amplitude drops out and only the geometry is left:

        score = mean|grad(G / max G)| * sqrt(fraction of cells above 0.15 max G)

    i.e. FINE (sharp gradients) and SPREAD OUT (a long meandering plume, detached
    vortices) rather than merely intense. Empirically it selects the snapshots that
    read best on a figure, and they sit at the BOTTOM of the variance ranking.

    Returns an np.ndarray of shape (n_snapshots,).
    """
    key = {"T": "u1_in", "kappa": "u2_in", "I": "I_target", "G": "I_target"}[field]
    X = dict_data[key].float()
    if field == "G":
        if n_dirs is None:
            n_dirs = int(torch.unique(dict_data["y"][:, 2]).shape[0])
        n_cells = X.shape[1] // n_dirs
        X = X.view(-1, n_cells, n_dirs).sum(dim=2) * dir_weight(n_dirs)
    if metric == "structure":
        g = int(round(math.sqrt(X.shape[1])))
        F = X.view(-1, g, g) / X.max(dim=1).values.clamp_min(1e-12)[:, None, None]
        gy, gx = np.gradient(F.numpy(), axis=(1, 2))
        sharp = np.sqrt(gx**2 + gy**2).reshape(F.shape[0], -1).mean(1)
        extent = (F > 0.15).float().view(F.shape[0], -1).mean(1).numpy()
        return sharp * np.sqrt(extent)
    var = X.var(dim=1, unbiased=False)
    if metric == "cv":
        return (var.sqrt() / X.mean(dim=1).abs().clamp_min(1e-12)).numpy()
    return var.numpy()


def most_chaotic_snapshots(dict_data, k=4, field="G", metric="var", n_dirs=None,
                           verbose=True):
    """Indices of the `k` MOST CHAOTIC snapshots of a split = largest spatial variance
    of the `field` (cf. `snapshot_variance`), sorted from most to least chaotic.

    This is the default selection for the comparison figures: a snapshot drawn at
    random is often flat and uninformative, whereas a high-variance field carries the
    structures (plume, fronts) where the models really differ.

    Returns (indices np.ndarray (k,), variances np.ndarray (n_snapshots,)).
    """
    v = snapshot_variance(dict_data, field=field, n_dirs=n_dirs, metric=metric)
    idx = np.argsort(v)[::-1][:k]
    if verbose:
        lbl = {"var": "variance", "cv": "coef. of variation",
               "structure": "structure score"}[metric]
        print(f"  {k} most chaotic snapshots (spatial {lbl} of {field}, "
              f"out of {len(v)} snapshots in the split):")
        for rank, i in enumerate(idx, 1):
            print(f"    #{rank}  snapshot {int(i):4d}   {lbl} = {v[i]:.4e}"
                  f"   (split median = {np.median(v):.4e})")
    return idx, v


class Evaluator:
    """
    Evaluation and visualisation of the model predictions.

    The encoder (PCA, autoencoder or Transformer-Based DeepONet) is attached through
    the `self.encoder` attribute after construction:
        evaluator = Evaluator(model, device)
        evaluator.encoder = encoder
    """

    def __init__(self, model, device,
                 h5_path=None):
        self.model = model
        self.device = device
        self.h5_path = h5_path or solver_h5()
        self.encoder = None
        self.n_dirs_data = None

        self.N_dirs = 32

        with h5py.File(self.h5_path, "r") as f:
            cell_centers = f["mesh/cell_centers"][:]
            self.x_flat = cell_centers[:, 0]
            self.y_flat = cell_centers[:, 1]
            self.N_cells = len(self.x_flat)

            self.X_grid = self.x_flat.reshape(150, 150)
            self.Y_grid = self.y_flat.reshape(150, 150)

    def ensure_grid(self, y_full, grid=150):
        """Rebuild the plotting grid from `y_full` when the loaded mesh is DEGENERATE.

        When the solver H5 is unreachable we fall back on mesh_stub.h5, whose
        `mesh/cell_centers` contains ONLY zeros; a contourf on that grid gives an empty
        figure. The real spatial coordinates live in y_full (= dict_data['y'] of the
        .pt), in cell-major order: y_full[::n_dirs, :2]. Returns True if the grid was
        replaced.
        """
        if np.ptp(self.x_flat) > 0 or np.ptp(self.y_flat) > 0:
            return False
        n_dirs = self._infer_direction_count(y_full)
        cell_xy = torch.as_tensor(y_full)[::n_dirs, :2].cpu().numpy()
        self.x_flat = cell_xy[:, 0]
        self.y_flat = cell_xy[:, 1]
        self.N_cells = len(self.x_flat)
        self.X_grid = self.x_flat.reshape(grid, grid)
        self.Y_grid = self.y_flat.reshape(grid, grid)
        return True

    def _encode_inputs(self, u1_raw, u2_raw):
        """Apply the encoder (PCA/AE) to raw fields and return the
        latent tensors ready for the branch nets, on the right device."""
        if self.encoder is None:
            raise RuntimeError("No encoder attached: set `evaluator.encoder = encoder`.")
        temp = {"u1_in": u1_raw, "u2_in": u2_raw}
        temp = self.encoder.transform_dicts(temp, ["u1_in", "u2_in"])
        return temp["u1_in"].to(self.device), temp["u2_in"].to(self.device)

    def calculate_relative_norms(self, pred_tensor, true_tensor):
        """Compute the relative L1 and L2 errors."""
        pred = torch.as_tensor(pred_tensor, dtype=torch.float32).flatten()
        true = torch.as_tensor(true_tensor, dtype=torch.float32).flatten()
        norm2_true = torch.norm(true, p=2)
        norm1_true = torch.norm(true, p=1)

        if norm2_true < 1e-8:
            return 0.0, 0.0

        l2_err = (torch.norm(pred - true, p=2) / norm2_true).item()
        l1_err = (torch.norm(pred - true, p=1) / norm1_true).item()
        return l2_err, l1_err

    def _infer_direction_count(self, y_coords):
        """Infer the number of angular directions from y = [x, y, phi]."""
        return int(torch.unique(y_coords[:, 2]).shape[0])

    def _integrate_over_directions(self, I_flat, n_dirs):
        """Integrate I over the directions to get G, with a uniform quadrature."""
        n_points = I_flat.shape[1]
        if n_points % n_dirs != 0:
            raise ValueError(
                f"I size ({n_points}) is not divisible by n_dirs ({n_dirs})."
            )
        n_cells = n_points // n_dirs
        I_reshaped = I_flat.view(-1, n_cells, n_dirs)
        return torch.sum(I_reshaped, dim=2) * dir_weight(n_dirs)

    def _predict_chunked(self, u_in, y_coords, chunk=20000):
        """Model forward over a grid of points, IN CHUNKS (bounded memory).

        Evaluates the trunk on blocks of `chunk` points instead of the whole grid at
        once: the peak memory no longer depends on the total number of points (which
        matters for 150*150*16 = 360000, where one trunk activation is 704 MB/layer).
        Returns (B, P), under no_grad. The order of the points is preserved.
        """
        y_coords = y_coords.to(self.device)
        preds = []
        with torch.no_grad():
            for k in range(0, y_coords.shape[0], chunk):
                out = self.model(u_in, y_coords[k:k + chunk])
                preds.append(out.unsqueeze(0) if out.dim() == 1 else out)
        return torch.cat(preds, dim=1)

    @staticmethod
    def _per_snapshot_norms(pred, true, eps=1e-8):
        """PER-SNAPSHOT relative L2 and L1 errors (norm over dim=1).

        The convention of Lu et al. 2026 (Table 1): one error per snapshot, and the
        statistics (mean/std/min/max) are taken over the snapshots. Unlike a global
        norm over the flattened batch, every snapshot weighs the same, bright or dark.
        Retourne deux tenseurs (B,).
        """
        diff = pred - true
        l2 = diff.norm(p=2, dim=1) / true.norm(p=2, dim=1).clamp_min(eps)
        l1 = diff.norm(p=1, dim=1) / true.norm(p=1, dim=1).clamp_min(eps)
        return l2, l1

    @staticmethod
    def _per_snapshot_abs(pred, true):
        """PER-SNAPSHOT errors in PHYSICAL UNITS (no normalisation).

        Returns (||pred - true||_2, mean|pred - true|) per snapshot, both (B,). The
        first one is the ||I_m - N(u_m)|| of Thm 3.4 (cf. lower_bound/), whose
        root-mean-square over the snapshots is E_hat -- the quantity the bound applies
        to. The second is a mean absolute error, in W/m2/sr for I and W/m2 for G.
        """
        diff = (pred - true).double()
        return diff.norm(p=2, dim=1), diff.abs().mean(dim=1)

    def evaluate_split_errors(self, loader, y_coords=None, label="TEST", g_ref=None):
        """PER-SNAPSHOT relative L1/L2 errors over the given DataLoader, aggregated
        into mean/std/min/max (the metric of Table 1 of Lu et al. 2026).

        `label` is only a display label: the split is whatever the loader holds.
        Callers that REPORT a performance pass the test split (cf. `split_data`).

        The loader must yield ALREADY ENCODED inputs (u1_in, u2_in latents).
        The I_l2/I_l1/G_l2/G_l1 keys of the returned dict are per-snapshot MEANS.

        `g_ref` (n_snapshots, n_cells), when given, replaces the integrated
        `sum_m I_target` as the REFERENCE for G -- pass `solver_G(splits[...])` to
        compare against the solver's own G field. It must follow the loader order
        (the loaders here are built with shuffle=False).

        The same pass also returns the errors in PHYSICAL UNITS (nothing is divided by
        the norm of the truth): `*_ehat` = RMS over the snapshots of ||pred - true||
        (the E_hat of `lower_bound/`, comparable with the Thm 3.4 bound), `*_rmse` =
        the same per point (E_hat / sqrt(P)), `*_mae` = mean |pred - true|.
        """
        self.model.eval()
        val_loader = loader
        l1_errors, l2_errors, g_l1_errors, g_l2_errors = [], [], [], []
        abs_I, mae_I, abs_G, mae_G = [], [], [], []
        n_pts_I = n_pts_G = None
        off = 0

        print(f"\n=== ERRORS OVER THE {label} SET ===")
        with torch.no_grad():
            for u1_in, u2_in, target_I in val_loader:
                u_in = [u1_in.to(self.device), u2_in.to(self.device)]
                target_I = target_I.to(self.device)

                pred_I = self._predict_chunked(u_in, y_coords)
                if target_I.dim() == 1:
                    target_I = target_I.unsqueeze(0)

                l2_snap, l1_snap = self._per_snapshot_norms(pred_I.cpu(), target_I.cpu())
                l2_errors.extend(l2_snap.tolist())
                l1_errors.extend(l1_snap.tolist())

                a2_snap, a1_snap = self._per_snapshot_abs(pred_I.cpu(), target_I.cpu())
                abs_I.extend(a2_snap.tolist())
                mae_I.extend(a1_snap.tolist())
                n_pts_I = pred_I.shape[1]

                if self.n_dirs_data is None:
                    self.n_dirs_data = self._infer_direction_count(y_coords)
                G_pred = self._integrate_over_directions(pred_I, self.n_dirs_data)
                if g_ref is None:
                    G_true = self._integrate_over_directions(target_I, self.n_dirs_data)
                else:
                    G_true = g_ref[off:off + pred_I.shape[0]].to(G_pred.device)
                off += pred_I.shape[0]
                g_l2_snap, g_l1_snap = self._per_snapshot_norms(G_pred.cpu(), G_true.cpu())
                g_l2_errors.extend(g_l2_snap.tolist())
                g_l1_errors.extend(g_l1_snap.tolist())

                ga2_snap, ga1_snap = self._per_snapshot_abs(G_pred.cpu(), G_true.cpu())
                abs_G.extend(ga2_snap.tolist())
                mae_G.extend(ga1_snap.tolist())
                n_pts_G = G_pred.shape[1]

        def _stats(errs):
            e = np.asarray(errs)
            return {"mean": e.mean(), "std": e.std(), "min": e.min(), "max": e.max()}

        stats_I, stats_G = _stats(l2_errors), _stats(g_l2_errors)
        mean_l1, mean_g_l1 = np.mean(l1_errors), np.mean(g_l1_errors)

        e_I, e_G = np.asarray(abs_I), np.asarray(abs_G)
        ehat_I, ehat_G = float(np.sqrt((e_I ** 2).mean())), float(np.sqrt((e_G ** 2).mean()))
        rmse_I, rmse_G = ehat_I / math.sqrt(n_pts_I), ehat_G / math.sqrt(n_pts_G)
        mae_I_m, mae_G_m = float(np.mean(mae_I)), float(np.mean(mae_G))

        print(f"   ({len(l2_errors)} snapshots, per-snapshot errors -- Table 1 format)")
        print(f"-> I  L2 rel. : mean={stats_I['mean']*100:.2f}%  std={stats_I['std']*100:.2f}%  "
              f"min={stats_I['min']*100:.2f}%  max={stats_I['max']*100:.2f}%")
        print(f"-> G  L2 rel. : mean={stats_G['mean']*100:.2f}%  std={stats_G['std']*100:.2f}%  "
              f"min={stats_G['min']*100:.2f}%  max={stats_G['max']*100:.2f}%")
        print(f"-> I  L1 rel. moyenne : {mean_l1*100:.2f}%   |   G  L1 rel. moyenne : {mean_g_l1*100:.2f}%")
        print("   --- ABSOLUTE errors (physical units, no normalisation) ---")
        print(f"-> I : E_hat = RMS_m||I - N(u)|| = {ehat_I:.4e}   RMSE/point = {rmse_I:.4e}   "
              f"MAE = {mae_I_m:.4e}   [W/m2/sr]")
        print(f"-> G : E_hat = RMS_m||G - G_pred|| = {ehat_G:.4e}   RMSE/point = {rmse_G:.4e}   "
              f"MAE = {mae_G_m:.4e}   [W/m2]")
        print("=======================================================\n")
        return {
            "I_l2": stats_I["mean"], "I_l1": mean_l1,
            "G_l2": stats_G["mean"], "G_l1": mean_g_l1,
            "I_l2_std": stats_I["std"], "I_l2_min": stats_I["min"], "I_l2_max": stats_I["max"],
            "G_l2_std": stats_G["std"], "G_l2_min": stats_G["min"], "G_l2_max": stats_G["max"],
            "I_ehat": ehat_I, "I_rmse": rmse_I, "I_mae": mae_I_m,
            "G_ehat": ehat_G, "G_rmse": rmse_G, "G_mae": mae_G_m,
            "n_snapshots": len(l2_errors), "n_points_I": int(n_pts_I), "n_points_G": int(n_pts_G),
        }

    def plot_dataset_snapshot(self, dict_data, snapshot_idx, output_dir="results/evaluation",
                              prefix="Train", err_mode="peak"):
        """
        Take one snapshot out of the PyTorch dict (RAW fields), encode the inputs,
        run the model, integrate over the directions to get G, and plot the comparison
        against the dataset ground truth:

            G truth  |  G model  |  RELATIVE error in % (cf. `rel_error_map`)

        The first two panels share the G scale (set by the truth, hence identical for
        every model on that snapshot).
        """
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)

        u1_raw = dict_data["u1_in"][snapshot_idx:snapshot_idx + 1]
        u2_raw = dict_data["u2_in"][snapshot_idx:snapshot_idx + 1]
        I_target_raw = dict_data["I_target"][snapshot_idx:snapshot_idx + 1]
        y_coords = dict_data["y"]
        self.ensure_grid(y_coords)

        u1_branch, u2_branch = self._encode_inputs(u1_raw, u2_raw)

        I_pred = self._predict_chunked([u1_branch, u2_branch], y_coords)

        I_target = I_target_raw.to(self.device)

        n_dirs = self._infer_direction_count(y_coords)
        G_pred = self._integrate_over_directions(I_pred, n_dirs).squeeze()
        G_target = self._integrate_over_directions(I_target, n_dirs).squeeze()

        l2_err, _ = self.calculate_relative_norms(G_pred.cpu(), G_target.cpu())

        G_pred_np = G_pred.cpu().numpy()
        G_target_np = G_target.cpu().numpy()
        err_G = rel_error_map(G_pred_np, G_target_np, mode=err_mode)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        g_min = float(G_target_np.min())
        g_max = float(G_target_np.max())
        g_levels = np.linspace(g_min, g_max, 21)

        im0 = axs[0].contourf(self.X_grid, self.Y_grid, G_target_np.reshape(150, 150),
                              levels=g_levels, cmap="jet", extend="both")
        axs[0].set_title(f"G truth ({prefix})")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].contourf(self.X_grid, self.Y_grid, G_pred_np.reshape(150, 150),
                              levels=g_levels, cmap="jet", extend="both")
        axs[1].set_title(f"G model ({prefix})\n$L_2$ rel = {l2_err*100:.2f}%")
        fig.colorbar(im1, ax=axs[1])

        norm_txt = "max(G truth)" if err_mode == "peak" else "local |G truth|"
        e_levels, _ = sym_error_levels([err_G])
        im2 = axs[2].contourf(self.X_grid, self.Y_grid, err_G.reshape(150, 150),
                              levels=e_levels, cmap=ERR_CMAP, extend="both")
        axs[2].set_title(_err_title(err_G, f"Signed relative error "
                                           f"(model - truth) / {norm_txt}"))
        fig.colorbar(im2, ax=axs[2], format="%+.1f%%")

        for ax in axs:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")

        plt.suptitle(f"Analyse {prefix} — snapshot {snapshot_idx} | "
                     f"relative $L_2$ error on G = {l2_err*100:.2f}%", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        save_path = os.path.join(output_dir, f"{prefix}_Snapshot_{snapshot_idx}.pdf")
        save_path = save_figure(save_path)
        plt.close()

    def plot_snapshot_comparison(self, snapshot_idx=0, output_dir="results/evaluation", div_key="divQr"):
        """
        Compare four reconstructions of the incident radiation G for one snapshot:
          - G_true        : the G field stored by the FVDOM solver (HDF5)
          - G_true_integ  : integral of the 16 dataset intensities I (uniform quadrature)
          - G_pred_16     : integral of the model over the 16 training directions
          - G_pred_gauss  : integral of the model over 32 Gauss-Legendre nodes
                            (generalisation to unseen angles)

        Every reconstruction uses the same normalisation sum(w) = 4pi (the OpenFOAM
        convention) and the repository's single angular convention (`dirs_from_phi`),
        so they are directly comparable with G_true. G_pred_16 therefore lands exactly
        on the 16 training labels (`ordinate_phis`).
        """
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)

        with h5py.File(self.h5_path, "r") as f:
            time_groups = sorted([key for key in f.keys() if key.startswith("time_")])
            t_grp = time_groups[min(snapshot_idx, len(time_groups) - 1)]
            print(f"\nGenerating the maps for snapshot: {t_grp}")
            kappa = f[f"{t_grp}/aLambda_0"][:]
            T = f[f"{t_grp}/T"][:]
            G_true = f[f"{t_grp}/G"][:]
            I_data = f[f"{t_grp}/I"][:]

        u1_sensors = torch.tensor(T, dtype=torch.float32).unsqueeze(0)
        u2_sensors = torch.tensor(kappa, dtype=torch.float32).unsqueeze(0)
        u1_branch, u2_branch = self._encode_inputs(u1_sensors, u2_sensors)

        x_t = torch.from_numpy(self.x_flat).float().to(self.device)
        y_t = torch.from_numpy(self.y_flat).float().to(self.device)

        def reconstruct_G(phis, weights):
            G = torch.zeros(self.N_cells, device=self.device)
            with torch.no_grad():
                for phi_m, w_m in zip(phis, weights):
                    phi_t = torch.full((self.N_cells,), float(phi_m), device=self.device)
                    trunk_input = torch.stack([x_t, y_t, phi_t], dim=1)
                    I_m = self.model([u1_branch, u2_branch], trunk_input).squeeze()
                    G = G + I_m * float(w_m)
            return G.cpu().numpy()

        n16 = I_data.shape[1]
        w16 = np.full(n16, dir_weight(n16))
        G_pred_16 = reconstruct_G(ordinate_phis(n16), w16)

        n32 = self.N_dirs
        w32 = np.full(n32, dir_weight(n32))
        G_pred_32u = reconstruct_G(ordinate_phis(n32), w32)

        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(self.N_dirs)
        phi_gl = math.pi * (gl_nodes + 1.0)
        w_gl = 2.0 * math.pi * gl_weights
        print(f"Gauss-Legendre integration of I -> G ({self.N_dirs} directions, sum(w) = {w_gl.sum():.4f} ~ 4pi)...")
        G_pred_gauss = reconstruct_G(phi_gl, w_gl)

        G_true_integ = I_data.sum(axis=1) * dir_weight(n16)

        def l2_rel(a):
            return float(np.linalg.norm(a - G_true) / (np.linalg.norm(G_true) or 1.0))
        err_integ = l2_rel(G_true_integ)
        err_16 = l2_rel(G_pred_16)
        err_32u = l2_rel(G_pred_32u)
        err_gauss = l2_rel(G_pred_gauss)
        print(f"   L2 rel vs G_true | ∫16 data = {err_integ*100:.2f}% | pred16 = {err_16*100:.2f}% "
              f"| pred32unif = {err_32u*100:.2f}% | predGauss = {err_gauss*100:.2f}%")

        fields = [
            (G_true,       "G_true (solver data)",      None),
            (G_true_integ, r"G_true_integrated ($\int$ 16 I data)", err_integ),
            (G_pred_16,    r"G_pred_16 ($\int$ 16 dir.)",    err_16),
            (G_pred_32u,   r"G_pred_32unif ($\int$ 32 unif.)", err_32u),
            (G_pred_gauss, r"G_pred_Gauss ($\int$ 32 GL)",   err_gauss),
        ]
        g_max = max(float(np.max(fld)) for fld, _, _ in fields)
        g_min = min(float(np.min(fld)) for fld, _, _ in fields)
        g_levels = np.linspace(g_min, g_max, 21)

        errs = [
            (rel_error_map(G_true_integ, G_true), r"$\int$16 data $-$ G_true"),
            (rel_error_map(G_pred_16, G_true),    "pred16 − G_true"),
            (rel_error_map(G_pred_32u, G_true),   "pred32unif − G_true"),
            (rel_error_map(G_pred_gauss, G_true), "predGauss − G_true"),
        ]
        e_levels, _ = sym_error_levels([e for e, _ in errs])

        fig, axs = plt.subplots(2, 5, figsize=(26, 10))
        for j, (field, title, err) in enumerate(fields):
            ttl = title if err is None else f"{title}\nL2 rel = {err*100:.2f}%"
            im = axs[0, j].contourf(self.X_grid, self.Y_grid, field.reshape(150, 150),
                                    levels=g_levels, cmap="jet", extend="both")
            axs[0, j].set_title(ttl, fontsize=11)
            fig.colorbar(im, ax=axs[0, j])

        axs[1, 0].axis("off")
        for k, (err_field, et) in enumerate(errs):
            im = axs[1, k + 1].contourf(self.X_grid, self.Y_grid, err_field.reshape(150, 150),
                                        levels=e_levels, cmap=ERR_CMAP, extend="both")
            axs[1, k + 1].set_title(_err_title(err_field, f"({et}) / max(G_true)"),
                                    fontsize=11)
            fig.colorbar(im, ax=axs[1, k + 1], format="%+.1f%%")

        for ax in axs.flat:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")

        plt.suptitle(f"Irradiance G — {t_grp} (Σω = 4π, signe trunk -cosφ/-sinφ)", fontsize=15)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"G_Comparison_{t_grp}.pdf")
        save_path = save_figure(save_path)
        plt.close()
        print(f" -> Figure saved to: {save_path}")

        rel_diff = rel_error_map(G_pred_16, G_true, signed=False)
        fig2, ax2 = g_simple_figure(G_true, G_pred_16, rel_diff,
                                    self.X_grid, self.Y_grid)
        save_path2 = os.path.join(output_dir, f"G_simple_{t_grp}.pdf")
        save_path2 = save_figure(save_path2)
        plt.close()
        print(f" -> Simple figure saved to: {save_path2}")

        return {
            "snapshot": t_grp,
            "integ": err_integ,
            "pred16": err_16,
            "pred32unif": err_32u,
            "predGauss": err_gauss,
        }

    def plot_snapshot_comparison_dataset(self, dict_data, snapshot_idx, y_full,
                                         output_dir="results/evaluation", include_unseen_angles=True,
                                         err_mode="peak", g_vmax=None):
        """Variant of `plot_snapshot_comparison` that does NOT DEPEND on the raw solver
        H5: T, kappa and I are sourced straight from `dict_data` (a split dict of the .pt).

        IMPORTANT: unlike the H5 version, the (x,y) coordinates AND the phi angles used
        as MODEL input come from `y_full` (the .pt), NOT from `self.x_flat/self.y_flat`
        (mesh_stub.h5) -- the latter only holds degenerate PLACEHOLDER coordinates
        (everything at ~[0,0], verified empirically), usable for the figure layout
        (X_grid/Y_grid) but never as model input. The 16 trained directions are READ
        from y_full (the source of truth); unseen angles go through `ordinate_phis`
        (cf. `dirs_from_phi` for the single angular convention).

        Difference with `plot_snapshot_comparison`: there is no independent SOLVER
        G_true field (it only lives in the H5), so the REFERENCE becomes
        G_ref = integral of I_data over the n_dirs stored directions (= the training
        target itself; in practice the most relevant comparison for judging the model,
        independently of solver/dataset fidelity).

        `include_unseen_angles=False` only shows and computes G_ref and G_pred_16 (the
        16 directions actually trained), dropping the unseen-angle generalisation panels
        (32 uniform, Gauss-Legendre) -- which also saves their cost (32+32 fewer decoder
        passes per snapshot)."""
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)

        T = dict_data["u1_in"][snapshot_idx].numpy()
        kappa = dict_data["u2_in"][snapshot_idx].numpy()
        I_flat = dict_data["I_target"][snapshot_idx].numpy()
        tag = f"dataset_snap{snapshot_idx}"
        print(f"\nGenerating the maps for snapshot (.pt, no solver H5): {tag}")

        u1_sensors = torch.tensor(T, dtype=torch.float32).unsqueeze(0)
        u2_sensors = torch.tensor(kappa, dtype=torch.float32).unsqueeze(0)
        u1_branch, u2_branch = self._encode_inputs(u1_sensors, u2_sensors)

        phis_true = torch.unique(y_full[:, 2])
        n16 = phis_true.shape[0]
        cell_xy = y_full[::n16, :2]
        n_cells = cell_xy.shape[0]
        x_t = cell_xy[:, 0].to(self.device)
        y_t = cell_xy[:, 1].to(self.device)

        def reconstruct_G(phis, weights):
            """sum_m w_m I_model(phi_m); phis = angles already built (read from y_full,
            or produced by `ordinate_phis`)."""
            G = torch.zeros(n_cells, device=self.device)
            with torch.no_grad():
                for phi_m, w_m in zip(phis, weights):
                    phi_t = torch.full((n_cells,), float(phi_m), device=self.device)
                    trunk_input = torch.stack([x_t, y_t, phi_t], dim=1)
                    I_m = self.model([u1_branch, u2_branch], trunk_input).squeeze()
                    G = G + I_m * float(w_m)
            return G.cpu().numpy()

        I_data = I_flat.reshape(n_cells, n16)

        w16 = np.full(n16, dir_weight(n16))
        G_pred_16 = reconstruct_G(phis_true.tolist(), w16)

        G_pred_32u = G_pred_gauss = err_32u = err_gauss = None
        if include_unseen_angles:
            n32 = self.N_dirs
            w32 = np.full(n32, dir_weight(n32))
            G_pred_32u = reconstruct_G(ordinate_phis(n32), w32)

            gl_nodes, gl_weights = np.polynomial.legendre.leggauss(self.N_dirs)
            phi_gl = math.pi * (gl_nodes + 1.0)
            w_gl = 2.0 * math.pi * gl_weights
            G_pred_gauss = reconstruct_G(phi_gl, w_gl)

        G_ref = I_data.sum(axis=1) * dir_weight(n16)

        X_grid = cell_xy[:, 0].numpy().reshape(150, 150)
        Y_grid = cell_xy[:, 1].numpy().reshape(150, 150)

        def l2_rel(a):
            return float(np.linalg.norm(a - G_ref) / (np.linalg.norm(G_ref) or 1.0))
        err_16 = l2_rel(G_pred_16)
        msg = (f"   G-L2 = {err_16*100:.2f} %   "
               f"max |dG| / max(G) = {100 * float(np.max(np.abs(G_pred_16 - G_ref))) / float(np.max(G_ref)):.2f} %")
        if include_unseen_angles:
            err_32u = l2_rel(G_pred_32u)
            err_gauss = l2_rel(G_pred_gauss)
            msg += f" | pred32unif = {err_32u*100:.2f}% | predGauss = {err_gauss*100:.2f}%"
        print(msg)

        fields = [
            (G_ref,        r"G_ref ($\int$ data, solveur indisponible)", None),
            (G_pred_16,    r"G_pred_16 ($\int$ 16 dir.)",    err_16),
        ]
        errs = [(rel_error_map(G_pred_16, G_ref, mode=err_mode),
                 "(pred16 − G_ref) / max(G_ref)")]
        if include_unseen_angles:
            fields += [
                (G_pred_32u,   r"G_pred_32unif ($\int$ 32 unif.)", err_32u),
                (G_pred_gauss, r"G_pred_Gauss ($\int$ 32 GL)",   err_gauss),
            ]
            errs += [
                (rel_error_map(G_pred_32u, G_ref, mode=err_mode),   "(pred32unif − G_ref) / max(G_ref)"),
                (rel_error_map(G_pred_gauss, G_ref, mode=err_mode), "(predGauss − G_ref) / max(G_ref)"),
            ]
        n_cols = len(fields)

        g_max = float(np.max(G_ref))
        g_min = float(np.min(G_ref))
        g_levels = np.linspace(g_min, g_max, 21)

        e_levels, _ = sym_error_levels([e for e, _ in errs])

        if include_unseen_angles:
            fig, axs = plt.subplots(2, n_cols, figsize=(5.25 * n_cols, 10), squeeze=False)
            for j, (field, title, err) in enumerate(fields):
                ttl = title if err is None else f"{title}\nL2 rel = {err*100:.2f}%"
                im = axs[0, j].contourf(X_grid, Y_grid, field.reshape(150, 150),
                                        levels=g_levels, cmap="jet", extend="both")
                axs[0, j].set_title(ttl, fontsize=11)
                fig.colorbar(im, ax=axs[0, j])

            axs[1, 0].axis("off")
            for k, (err_field, et) in enumerate(errs):
                im = axs[1, k + 1].contourf(X_grid, Y_grid, err_field.reshape(150, 150),
                                            levels=e_levels, cmap=ERR_CMAP, extend="both")
                axs[1, k + 1].set_title(_err_title(err_field, et), fontsize=11)
                fig.colorbar(im, ax=axs[1, k + 1], format="%+.1f%%")
            for j in range(len(errs) + 1, n_cols):
                axs[1, j].axis("off")

            for ax in axs.flat:
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_aspect("equal")

            plt.suptitle(f"Incident radiation G -- {tag} (sum(w) = 4pi, ref = int data, solver H5 "
                         f"indisponible)", fontsize=14)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"G_Comparison_{tag}.pdf")
            save_path = save_figure(save_path)
            plt.close()
            print(f" -> Figure saved to: {save_path}")

        rel_diff = rel_error_map(G_pred_16, G_ref, signed=False)
        fig2, ax2 = g_simple_figure(G_ref, G_pred_16, rel_diff, X_grid, Y_grid,
                                    g_vmax=g_vmax)
        save_path2 = os.path.join(output_dir, f"G_simple_{tag}.pdf")
        save_path2 = save_figure(save_path2)
        plt.close()
        print(f" -> Simple figure saved to: {save_path2}")

        return {"snapshot": tag, "pred16": err_16, "pred32unif": err_32u, "predGauss": err_gauss}


def build_collocation(split_data, y_full, n_dirs, n_cells, trained_dirs, seed, device):
    """Build the (x, y, phi) points covering the domain, plus the cell indices.

    The spatial points ARE mesh cell centres, so the source S = kappa*(sigma/pi)*T^4
    is EXACT there. The directions are either the 16 trained ones (--trained_dirs) or
    n_dirs random continuous phi angles over [0, 2pi).

    Returns (y (P,3), cell_idx (P,), T (N,ncells), kappa (N,ncells)).
    """
    n_dirs_data = int(torch.unique(y_full[:, 2]).shape[0])
    cell_xy = y_full[::n_dirs_data, :2]
    n_cells_total = cell_xy.shape[0]

    g = torch.Generator().manual_seed(seed)
    if n_cells is not None and n_cells < n_cells_total:
        sel_cells = torch.randperm(n_cells_total, generator=g)[:n_cells]
    else:
        sel_cells = torch.arange(n_cells_total)
    cell_xy_sel = cell_xy[sel_cells]

    C = cell_xy_sel.shape[0]
    if trained_dirs:
        phis = torch.unique(y_full[:, 2])
        n_d = phis.shape[0]
        xy_rep = cell_xy_sel.repeat_interleave(n_d, dim=0)
        phi_rep = phis.repeat(C)
        desc = f"{n_d} TRAINED directions (the data angles)"
    else:
        n_d = n_dirs
        xy_rep = cell_xy_sel.repeat_interleave(n_d, dim=0)
        phi_rep = torch.rand(C * n_d, generator=g) * (2 * math.pi)
        desc = f"{n_d} RANDOM continuous directions per cell"
    y = torch.cat([xy_rep, phi_rep.unsqueeze(1)], dim=1).to(device)
    cell_idx = sel_cells.repeat_interleave(n_d)

    T = split_data["u1_in"].float()
    kappa = split_data["u2_in"].float()
    print(f"Collocation : {C} cellules × {n_d} directions = {y.shape[0]} points ({desc}).")
    return y, cell_idx, T, kappa


def evaluate(model, encoder, y, cell_idx, T, kappa, device,
             n_snapshots=None, chunk=64, reservoir_cap=2_000_000, seed=0):
    """Compute the residual r = mu.grad(I) + kappa*I - S over every (snapshot, point).

    Derivatives come from model.pde_residual (exact autodiff). Online accumulators
    (RMS, correlation) plus a reservoir of |r| for percentiles/histogram.
    """
    from deeponet import IB_COEF, INPUT_KEYS

    enc = encoder.transform_dicts({"u1_in": T, "u2_in": kappa}, INPUT_KEYS)
    u1_enc = enc["u1_in"].to(device)
    u2_enc = enc["u2_in"].to(device)
    cell_idx = cell_idx.to(device)
    T, kappa = T.to(device), kappa.to(device)

    n_snap = u1_enc.shape[0] if n_snapshots is None else min(u1_enc.shape[0], n_snapshots)
    print(f"\nEvaluating on {n_snap} snapshots x {y.shape[0]} points "
          f"= {n_snap * y.shape[0]:,} points...")

    acc = {k: 0.0 for k in ("n", "r2", "abs_r", "max_r", "S2",
                            "tr", "rhs", "tr2", "rhs2", "tr_rhs")}
    per_snap_rel = []
    reservoir = []
    rng = np.random.default_rng(seed)

    for s0 in range(0, n_snap, chunk):
        s1 = min(s0 + chunk, n_snap)
        T_c = T[s0:s1][:, cell_idx]
        k_c = kappa[s0:s1][:, cell_idx]
        S = k_c * IB_COEF * T_c ** 4
        u1_eval = S
        u2_eval = -k_c

        res, I = model.pde_residual([u1_enc[s0:s1], u2_enc[s0:s1]], y, u1_eval, u2_eval)
        res, I = res.detach(), I.detach()
        rhs = u1_eval + u2_eval * I
        transport = res + rhs

        acc["n"] += res.numel()
        acc["r2"] += float((res.double() ** 2).sum())
        acc["abs_r"] += float(res.abs().double().sum())
        acc["max_r"] = max(acc["max_r"], float(res.abs().max()))
        acc["S2"] += float((S.double() ** 2).sum())
        acc["tr"] += float(transport.double().sum())
        acc["rhs"] += float(rhs.double().sum())
        acc["tr2"] += float((transport.double() ** 2).sum())
        acc["rhs2"] += float((rhs.double() ** 2).sum())
        acc["tr_rhs"] += float((transport.double() * rhs.double()).sum())

        rms_r_b = torch.sqrt((res ** 2).mean(dim=1))
        rms_S_b = torch.sqrt((S ** 2).mean(dim=1)).clamp_min(1e-30)
        per_snap_rel.append((rms_r_b / rms_S_b).cpu())

        flat = res.abs().reshape(-1).cpu().numpy()
        if reservoir_cap and flat.size > 0:
            take = min(flat.size, max(1, reservoir_cap // n_snap + 1))
            reservoir.append(flat[rng.integers(0, flat.size, size=take)])

    acc["per_snap_rel"] = torch.cat(per_snap_rel).numpy()
    acc["reservoir"] = np.concatenate(reservoir) if reservoir else np.array([0.0])
    return acc


def report(acc, model_tag, i_max=None):
    """Print the physics verdict. When i_max (= max I_target, a FIXED scale) is given,
    also report the physics loss normalised by the intensity peak."""
    n = int(acc["n"])
    rms_r = math.sqrt(acc["r2"] / n)
    mean_abs = acc["abs_r"] / n
    rms_S = math.sqrt(acc["S2"] / n)
    rms_tr = math.sqrt(acc["tr2"] / n)
    rms_rhs = math.sqrt(acc["rhs2"] / n)
    rho = rms_r / max(rms_S, 1e-30)

    mean_tr, mean_rhs = acc["tr"] / n, acc["rhs"] / n
    cov = acc["tr_rhs"] / n - mean_tr * mean_rhs
    std_tr = math.sqrt(max(acc["tr2"] / n - mean_tr ** 2, 0.0))
    std_rhs = math.sqrt(max(acc["rhs2"] / n - mean_rhs ** 2, 0.0))
    corr = cov / max(std_tr * std_rhs, 1e-30)

    p50, p90, p99 = np.percentile(acc["reservoir"], [50, 90, 99])
    rel = acc["per_snap_rel"]

    verdict = ("PDE very well satisfied" if rho < 0.1 else
               "PDE partly satisfied" if rho < 0.5 else
               "PDE barely satisfied")

    print(f"\n=========== 2D RTE PHYSICS RESIDUAL -- {model_tag} ===========")
    print(f"|  Equation:  mu.grad(I) + kappa*I = S   (S = kappa*(sigma/pi)*T^4 = forcing)")
    print(f"|  Points evaluated: {n:,}")
    print("|  ---------------------------------------------------------")
    print(f"|  >> RELATIVE RESIDUAL  rho = RMS(r)/RMS(S) = {rho:.4f}  ({100 * rho:.1f} %)")
    print(f"|     -> the PDE is violated by {100 * rho:.1f} % of the forcing")
    print(f"|     -> {verdict}   (reference: <10 % good, ~100 % not satisfied)")
    if i_max is not None and i_max > 0:
        rms_over_imax = rms_r / i_max
        loss_I = rms_over_imax ** 2
        print(f"|  >> PHYS LOSS / max(I):  RMS(r)/max(I) = {rms_over_imax:.4f}  ({100 * rms_over_imax:.2f} %)")
        print(f"|     mean(r^2)/max(I)^2 = {loss_I:.4e}     max(I)_data = {i_max:.4e} (FIXED scale)")
    print(f"|  >> CORRELATION  mu.grad(I) vs (S-kappa*I) = {corr:+.3f}")
    print(f"|     -> 1 = transport explains the physics | 0 = transport is noise")
    print("|  ---------------------------------------------------------")
    print(f"|  RMS residual    RMS(r)   = {rms_r:.4e}")
    print(f"|  RMS source      RMS(S)   = {rms_S:.4e}   (FIXED scale, data only)")
    print(f"|  RMS transport   RMS(mu.grad(I))= {rms_tr:.4e}")
    print(f"|  RMS right-hand side RMS(S-kappa*I)= {rms_rhs:.4e}")
    print(f"|  |r|: mean={mean_abs:.3e}  max={acc['max_r']:.3e}  "
          f"p50={p50:.3e} p90={p90:.3e} p99={p99:.3e}")
    print(f"|  rho per snapshot: min={rel.min():.3f}  mean={rel.mean():.3f}  max={rel.max():.3f}")
    print("╚════════════════════════════════════════════════════════════\n")
    return rho, corr


def check_autodiff(model, encoder, y, cell_idx, T, kappa, device, n_pts=2000):
    """Compare the transport term of pde_residual (jacfwd) with a direct
    torch.autograd.grad -> proves the derivative is exact autodiff, not a finite
    difference."""
    from deeponet import IB_COEF, INPUT_KEYS

    sel = torch.randperm(y.shape[0])[:min(n_pts, y.shape[0])]
    y_s = y[sel].to(device)
    ci = cell_idx[sel].to(device)
    enc = encoder.transform_dicts({"u1_in": T[:1], "u2_in": kappa[:1]}, INPUT_KEYS)
    u1b, u2b = enc["u1_in"].to(device), enc["u2_in"].to(device)
    Tc = T[:1].to(device)[:, ci]
    kc = kappa[:1].to(device)[:, ci]
    u1_eval, u2_eval = kc * IB_COEF * Tc ** 4, -kc

    res, I = model.pde_residual([u1b, u2b], y_s, u1_eval, u2_eval)
    transport_fast = (res + (u1_eval + u2_eval * I)).detach()

    y_req = y_s.clone().requires_grad_(True)
    I_ref = model([u1b, u2b], y_req)
    grad = torch.autograd.grad(I_ref.sum(), y_req)[0]
    sin_phi, cos_phi = transport_coeffs(y_s[:, 2])
    transport_ref = (sin_phi * grad[:, 0] + cos_phi * grad[:, 1]).detach()

    diff = (transport_fast.squeeze(0) - transport_ref).abs()
    rel = diff.max() / (transport_ref.abs().max() + 1e-30)
    print("\n--- Autodiff check (jacfwd vs torch.autograd.grad) ---")
    print(f"  max|delta transport| = {diff.max().item():.3e}  (relative = {rel.item():.3e})")
    print("  OK identical -> the derivative is exact autodiff (no finite difference)\n"
          if rel.item() < 1e-3 else "  WARNING unexpected gap, worth investigating\n")


def plot_residual_map(model, encoder, split_data, y_full, snapshot_idx, phi,
                      device, output_dir, grid=(150, 150)):
    """Spatial map of the predicted I and of |PDE residual| for one snapshot and one
    direction phi (angle from +y clockwise, cf. `dirs_from_phi`)."""
    from deeponet import IB_COEF, INPUT_KEYS

    os.makedirs(output_dir, exist_ok=True)
    n_dirs_data = int(torch.unique(y_full[:, 2]).shape[0])
    x_cell = y_full[::n_dirs_data, 0].numpy()
    y_cell = y_full[::n_dirs_data, 1].numpy()
    n_cells = x_cell.shape[0]

    T = split_data["u1_in"][snapshot_idx].float().to(device)
    kappa = split_data["u2_in"][snapshot_idx].float().to(device)
    u1_eval = (kappa * IB_COEF * T ** 4).unsqueeze(0)
    u2_eval = (-kappa).unsqueeze(0)

    enc = encoder.transform_dicts(
        {"u1_in": split_data["u1_in"][snapshot_idx:snapshot_idx + 1],
         "u2_in": split_data["u2_in"][snapshot_idx:snapshot_idx + 1]}, INPUT_KEYS)
    u1b, u2b = enc["u1_in"].to(device), enc["u2_in"].to(device)

    x_t = torch.tensor(x_cell, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_cell, dtype=torch.float32, device=device)
    phi_t = torch.full((n_cells,), float(phi), device=device)
    y_trunk = torch.stack([x_t, y_t, phi_t], dim=1)

    res, I = model.pde_residual([u1b, u2b], y_trunk, u1_eval, u2_eval)
    res_np = res.detach().reshape(grid).cpu().numpy()
    I_np = I.detach().reshape(grid).cpu().numpy()
    X, Y = x_cell.reshape(grid), y_cell.reshape(grid)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axs[0].contourf(X, Y, I_np, levels=50, cmap="jet")
    axs[0].set_title(f"predicted I (snapshot {snapshot_idx}, phi={phi:.2f})")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].contourf(X, Y, np.abs(res_np), levels=50, cmap="Reds")
    axs[1].set_title("|PDE residual|")
    fig.colorbar(im1, ax=axs[1])
    for ax in axs:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_aspect("equal")
    rms = float(np.sqrt(np.mean(res_np ** 2)))
    plt.suptitle(f"2D RTE residual -- snapshot {snapshot_idx} | RMS = {rms:.4e}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f"residual_map_snap{snapshot_idx}_phi{phi:.2f}.pdf")
    path = save_figure(path)
    plt.close()
    print(f"Residual map saved to: {path}  (RMS = {rms:.4e})")


def plot_histogram(reservoir, output_dir, tag):
    """log10|r| histogram of the PDE residual."""
    os.makedirs(output_dir, exist_ok=True)
    a = reservoir[reservoir > 0]
    if a.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(np.log10(a), bins=80, color="steelblue", alpha=0.8)
    plt.xlabel(r"$\log_{10}|r|$"); plt.ylabel("Number of points")
    plt.title(f"PDE residual distribution ({tag})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(output_dir, f"residual_hist_{tag}.pdf")
    path = save_figure(path); plt.close()
    print(f"Residual histogram saved to: {path}")


def fd_residual_stats(I_t, T, kap, phis, dx, dy, mask=None):
    """Accumulate RMS(r), RMS(S) and the transport-vs-(S-kappa*I) correlation over
    every (snap, cell, dir), with grad(I) from finite differences (np.gradient).
    `phis`: angles (n_dir,)."""
    from deeponet import IB_COEF

    G = int(round(T.shape[1] ** 0.5))
    phis = np.asarray(phis, dtype=np.float64)
    n_snap, n_dir = I_t.shape[0], phis.shape[0]
    d = dirs_from_phi(phis)
    sin_phi = d[:, 0][None, None, :]
    cos_phi = d[:, 1][None, None, :]
    acc = {k: 0.0 for k in ("n", "r2", "S2", "tr", "rhs", "tr2", "rhs2", "tr_rhs", "maxI")}
    per_snap = []
    interior = (slice(1, -1), slice(1, -1), slice(None))
    for s in range(n_snap):
        I = I_t[s].reshape(G, G, n_dir).astype(np.float64)
        gy, gx = np.gradient(I, dy, dx, axis=(0, 1))
        transport = sin_phi * gx + cos_phi * gy
        Tg = T[s].reshape(G, G, 1).astype(np.float64)
        kg = kap[s].reshape(G, G, 1).astype(np.float64)
        S = np.broadcast_to(kg * IB_COEF * Tg ** 4, I.shape)
        rhs = S - kg * I
        r = transport - rhs
        if mask == "interior":
            r, S, transport, rhs, I = (a[interior] for a in (r, S, transport, rhs, I))
        acc["n"] += r.size
        acc["r2"] += (r ** 2).sum();   acc["S2"] += (S ** 2).sum()
        acc["tr"] += transport.sum();  acc["rhs"] += rhs.sum()
        acc["tr2"] += (transport ** 2).sum(); acc["rhs2"] += (rhs ** 2).sum()
        acc["tr_rhs"] += (transport * rhs).sum()
        acc["maxI"] = max(acc["maxI"], float(np.abs(I).max()))
        per_snap.append(math.sqrt((r ** 2).mean()) / max(math.sqrt((S ** 2).mean()), 1e-30))
    return acc, np.array(per_snap)


def report_fd(acc, per_snap, i_max_data, tag):
    """Verdict on the FD residual (raw FVDOM data)."""
    n = acc["n"]
    rms_r = math.sqrt(acc["r2"] / n)
    rms_S = math.sqrt(acc["S2"] / n)
    rho = rms_r / rms_S
    mtr, mrhs = acc["tr"] / n, acc["rhs"] / n
    cov = acc["tr_rhs"] / n - mtr * mrhs
    std_tr = math.sqrt(max(acc["tr2"] / n - mtr ** 2, 0.0))
    std_rhs = math.sqrt(max(acc["rhs2"] / n - mrhs ** 2, 0.0))
    corr = cov / max(std_tr * std_rhs, 1e-30)
    print(f"\n====== FD RESIDUAL OF THE FVDOM DATA -- {tag} ======")
    print(f"|  Points: {int(n):,}  (the 16 discrete data directions)")
    print(f"|  >> rho = RMS(r)/RMS(S)        = {rho:.4f}  ({100*rho:.1f} %)")
    print(f"|  >> RMS(r)/max(I)_data        = {rms_r/i_max_data:.4f}  ({100*rms_r/i_max_data:.2f} %)")
    print(f"|  >> CORRELATION mu.grad(I) vs (S-kappa*I) = {corr:+.3f}")
    print(f"|  RMS(r) = {rms_r:.4e}   RMS(S) = {rms_S:.4e}   max(I) = {i_max_data:.4e}")
    print(f"|  rho per snapshot: min={per_snap.min():.3f} mean={per_snap.mean():.3f} max={per_snap.max():.3f}")
    print("====================================================\n")


def fd_metrics_from_acc(acc, i_max):
    """rho = RMS(r)/RMS(S), RMS(r)/max(I), transport-vs-(S-kappa*I) correlation."""
    n = acc["n"]
    rms_r = math.sqrt(acc["r2"] / n)
    rms_S = math.sqrt(acc["S2"] / n)
    rho = rms_r / rms_S
    mtr, mrhs = acc["tr"] / n, acc["rhs"] / n
    cov = acc["tr_rhs"] / n - mtr * mrhs
    std_tr = math.sqrt(max(acc["tr2"] / n - mtr ** 2, 0.0))
    std_rhs = math.sqrt(max(acc["rhs2"] / n - mrhs ** 2, 0.0))
    corr = cov / max(std_tr * std_rhs, 1e-30)
    return rho, rms_r / i_max, corr


def fd_accumulate(acc_tot, acc):
    """Sum the fd_residual_stats accumulators over blocks of snapshots."""
    if not acc_tot:
        return dict(acc)
    for k in ("n", "r2", "S2", "tr", "rhs", "tr2", "rhs2", "tr_rhs"):
        acc_tot[k] += acc[k]
    acc_tot["maxI"] = max(acc_tot["maxI"], acc["maxI"])
    return acc_tot


@torch.no_grad()
def model_fd_residual(model, encoder, T, kap, phis, dx, dy, y_full, i_max,
                      ncell, n_dir, device, snap_block=64):
    """Predict the model I over y_full (ordered grid), then apply the SAME finite
    differences as on the FVDOM data -> an apples-to-apples residual. Returns
    (acc_tot, per_snap)."""
    from deeponet import INPUT_KEYS

    enc = encoder.transform_dicts({"u1_in": T, "u2_in": kap}, INPUT_KEYS)
    u1 = enc["u1_in"].to(device)
    u2 = enc["u2_in"].to(device)
    y = y_full.float().to(device)
    n_snap = u1.shape[0]
    acc_tot, per_snap = {}, []
    for s0 in range(0, n_snap, snap_block):
        s1 = min(s0 + snap_block, n_snap)
        I = model([u1[s0:s1], u2[s0:s1]], y).cpu().numpy()
        I = I.reshape(s1 - s0, ncell, n_dir)
        acc, ps = fd_residual_stats(I, T[s0:s1].numpy(), kap[s0:s1].numpy(),
                                    phis, dx, dy, mask=None)
        acc_tot = fd_accumulate(acc_tot, acc)
        per_snap.append(ps)
    return acc_tot, np.concatenate(per_snap)


FIELD_LABELS = {
    "u1_in": "u1_in (temperature T)",
    "u2_in": "u2_in (absorption kappa)",
}


def _rel_l2(orig, rec):
    """Global relative L2 between two flattened arrays."""
    return float(np.linalg.norm(orig - rec) / (np.linalg.norm(orig) or 1.0))


def reconstruct_and_plot(encoder, dict_data, field_key, snapshot_idx, encoder_kind,
                         grid_shape=(150, 150), output_dir=None,
                         raw_idx=None, split="all"):
    disp = snapshot_idx if raw_idx is None else raw_idx
    output_dir = output_dir or os.path.join(RESULTS_DIR, "figures")

    x_original = dict_data[field_key][snapshot_idx:snapshot_idx + 1].float()

    x_rec = encoder.reconstruct_field(field_key, x_original)

    x_orig_np = x_original.cpu().numpy().reshape(grid_shape)
    x_rec_np = x_rec.cpu().numpy().reshape(grid_shape)
    abs_error = np.abs(x_orig_np - x_rec_np)

    l2_rel = np.linalg.norm(x_orig_np - x_rec_np) / (np.linalg.norm(x_orig_np) or 1.0)
    max_err = abs_error.max()
    latent_d = encoder.latent_dims.get(field_key, "?")

    print(f"\n--- {encoder_kind.upper()} | {field_key} | split={split} | snapshot brut #{disp} ---")
    print(f"Dimension latente d = {latent_d}")
    print(f"Global relative L2 error : {l2_rel:.4e} ({l2_rel*100:.2f}%)")
    print(f"Maximum absolute error   : {max_err:.4e}")

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    label = FIELD_LABELS.get(field_key, field_key)

    im0 = axs[0].imshow(x_orig_np, cmap="magma", origin="lower")
    axs[0].set_title(f"Original {label}")
    map_colorbar(fig, im0, axs[0])

    im1 = axs[1].imshow(x_rec_np, cmap="magma", origin="lower")
    axs[1].set_title(f"Reconstruit ({encoder_kind}, d={latent_d}, split={split})")
    map_colorbar(fig, im1, axs[1])

    im2 = axs[2].imshow(abs_error, cmap="Reds", origin="lower")
    axs[2].set_title(r"Absolute error $|X - \hat{X}|$")
    map_colorbar(fig, im2, axs[2])

    for ax in axs:
        ax.axis("off")

    plt.suptitle(
        f"Reconstruction {encoder_kind} — {field_key} (split={split}, snapshot brut #{disp})\n"
        f"Relative $L_2$ error = {l2_rel:.4e} ({l2_rel*100:.2f}%)",
        fontsize=14,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"recon_{encoder_kind}_{field_key}_{split}_snap{disp}.pdf")
    save_path = save_figure(save_path)
    plt.close()
    print(f"-> Figure saved to: {save_path}")


def aggregate_split_error(encoder, dict_data, field_key, encoder_kind, split="val"):
    """Reconstruction error over the WHOLE split (not a single snapshot).

    Rebuilds every snapshot of the split in one pass and reports the MEAN per-snapshot
    relative L2 (directly comparable with the single-snapshot mode, and with the I/G
    errors of the model, which are per-snapshot means too), its standard deviation and
    its max over the snapshots. Every snapshot weighs the same.
    """
    X = dict_data[field_key].float()
    X_rec = encoder.reconstruct_field(field_key, X).float()
    n = X.shape[0]

    num = torch.linalg.norm(X - X_rec, dim=1)
    den = torch.linalg.norm(X, dim=1).clamp_min(1e-12)
    rel = num / den
    rel_mean, rel_std, rel_max = rel.mean().item(), rel.std().item(), rel.max().item()
    latent_d = encoder.latent_dims.get(field_key, "?")

    print(f"\n--- {encoder_kind.upper()} | {field_key} | split={split} | {n} snapshots | d={latent_d} ---")
    print(f"L2 rel moyenne / snapshot : {rel_mean*100:.2f}%  (±{rel_std*100:.2f}%, max {rel_max*100:.2f}%)")
    return {"field": field_key, "n": n, "latent_d": latent_d,
            "mean": rel_mean, "std": rel_std, "max": rel_max}


def reconstruct_and_plot_joint(encoder, dict_data, snapshot_idx, grid_shape=(150, 150),
                               output_dir=None, raw_idx=None, split="all"):
    """Plot orig/recon/error for BOTH fields of one snapshot (a 2x3 figure), using a
    joint encoder (a single latent z for (T, kappa) -> two decodings)."""
    disp = snapshot_idx if raw_idx is None else raw_idx
    output_dir = output_dir or os.path.join(RESULTS_DIR, "figures")

    u1 = dict_data["u1_in"][snapshot_idx:snapshot_idx + 1].float()
    u2 = dict_data["u2_in"][snapshot_idx:snapshot_idx + 1].float()
    u1_rec, u2_rec = encoder.reconstruct_fields(u1, u2)

    originals = {"u1_in": u1, "u2_in": u2}
    recons = {"u1_in": u1_rec, "u2_in": u2_rec}

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    print(f"\n--- JOINT_AE | split={split} | snapshot brut #{disp} | latent d={encoder.latent_dim} ---")

    for row, field_key in enumerate(["u1_in", "u2_in"]):
        x_orig = originals[field_key].cpu().numpy().reshape(grid_shape)
        x_rec = recons[field_key].cpu().numpy().reshape(grid_shape)
        abs_error = np.abs(x_orig - x_rec)
        l2_rel = _rel_l2(x_orig, x_rec)
        label = FIELD_LABELS.get(field_key, field_key)

        print(f"  {field_key:6s} | L2 rel = {l2_rel*100:.2f}%  | err abs max = {abs_error.max():.4e}")

        im0 = axs[row, 0].imshow(x_orig, cmap="magma", origin="lower")
        axs[row, 0].set_title(f"Original {label}")
        map_colorbar(fig, im0, axs[row, 0])

        im1 = axs[row, 1].imshow(x_rec, cmap="magma", origin="lower")
        axs[row, 1].set_title(f"Reconstruit (joint, d={encoder.latent_dim})")
        map_colorbar(fig, im1, axs[row, 1])

        im2 = axs[row, 2].imshow(abs_error, cmap="Reds", origin="lower")
        axs[row, 2].set_title(rf"Error $|X-\hat X|$ — rel. $L_2$ {l2_rel*100:.2f}%")
        map_colorbar(fig, im2, axs[row, 2])

    for ax in axs.flat:
        ax.axis("off")

    plt.suptitle(
        f"Reconstruction autoencodeur JOINT — split={split}, snapshot brut #{disp}",
        fontsize=14,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"recon_joint_{split}_snap{disp}.pdf")
    save_path = save_figure(save_path)
    plt.close()
    print(f"-> Figure saved to: {save_path}")


def aggregate_split_error_joint(encoder, dict_data, split="val"):
    """Mean per-snapshot relative L2 over the WHOLE split (joint encoder, both fields)."""
    u1 = dict_data["u1_in"].float()
    u2 = dict_data["u2_in"].float()
    u1_rec, u2_rec = encoder.reconstruct_fields(u1, u2)

    print(f"\n--- JOINT_AE | split={split} | {u1.shape[0]} snapshots | latent d={encoder.latent_dim} ---")
    out = []
    for field_key, X, X_rec in (("u1_in", u1, u1_rec), ("u2_in", u2, u2_rec)):
        num = torch.linalg.norm(X - X_rec, dim=1)
        den = torch.linalg.norm(X, dim=1).clamp_min(1e-12)
        rel = num / den
        print(f"  {field_key:6s} | L2 rel moyenne/snapshot : {rel.mean()*100:.2f}% "
              f"(±{rel.std()*100:.2f}%, max {rel.max()*100:.2f}%)")
        out.append({"field": field_key, "n": X.shape[0], "latent_d": encoder.latent_dim,
                    "mean": rel.mean().item(), "std": rel.std().item(),
                    "max": rel.max().item()})
    return out


def encoder_reconstruct_pair(encoder, u1, u2):
    """(T, kappa) rebuilt by any encoder: joint ones decode both from a single latent,
    the others field by field."""
    if hasattr(encoder, "reconstruct_fields"):
        return encoder.reconstruct_fields(u1, u2)
    return (encoder.reconstruct_field("u1_in", u1),
            encoder.reconstruct_field("u2_in", u2))


def encoder_split_error(encoder, dict_data, kind, split="test"):
    """Reconstruction error of an encoder over a whole split, for BOTH fields, whatever
    its kind -- returns [stats(T), stats(kappa)] as `aggregate_split_error` does."""
    if hasattr(encoder, "reconstruct_fields"):
        return aggregate_split_error_joint(encoder, dict_data, split=split)
    return [aggregate_split_error(encoder, dict_data, f, kind, split=split)
            for f in ("u1_in", "u2_in")]


def plot_encoder_comparison(entries, dict_data, snapshot_idx, grid_shape=(150, 150),
                            output_dir=None, raw_idx=None, split="all"):
    """Compare several encoders on ONE snapshot, bare: two rows (T, kappa), the solver
    field, then every reconstruction on the truth colour scale, then every
    |Delta| / ||truth||_inf in % on a scale shared across encoders. No title and no
    number on the panels -- each encoder is named by its family and its latent dimension,
    the errors live in the metrics files.

    `entries` = [(label, encoder), ...], e.g. [("PCA, d = 500", enc_pca),
    ("AE, d = 500", enc_ae)].
    """
    disp = snapshot_idx if raw_idx is None else raw_idx
    output_dir = output_dir or os.path.join(RESULTS_DIR, "figures")

    u1 = dict_data["u1_in"][snapshot_idx:snapshot_idx + 1].float()
    u2 = dict_data["u2_in"][snapshot_idx:snapshot_idx + 1].float()
    recs = [encoder_reconstruct_pair(enc, u1, u2) for _, enc in entries]

    # Every map is given a colour-bar slot of its own, left empty when it carries no
    # bar: all the panels keep exactly the same size, and the two bars that are drawn
    # take the height of the maps instead of that of their grid cell.
    n = len(entries)
    fig, axs = plt.subplots(2, 2 * n + 1, figsize=(3.4 * (2 * n + 1), 6.4), squeeze=False)
    caxs = [[make_axes_locatable(axs[r, c]).append_axes("right", size="4%", pad=0.10)
             for c in range(2 * n + 1)] for r in range(2)]
    print(f"\n--- reconstruction comparison | split={split} | snapshot brut #{disp} ---")

    for row, (x, sym, name) in enumerate(((u1, "T", "T"), (u2, r"\kappa", "kappa"))):
        truth = x.cpu().numpy().reshape(grid_shape)
        rec_np = [r[row].detach().cpu().numpy().reshape(grid_shape) for r in recs]
        vmin, vmax = float(truth.min()), float(truth.max())
        scale = float(np.abs(truth).max()) or 1.0
        errs = [100.0 * np.abs(truth - r) / scale for r in rec_np]
        emax = max(float(e.max()) for e in errs)

        im_f = axs[row, 0].imshow(truth, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
        axs[row, 0].set_ylabel(rf"${sym}$", fontsize=15)

        for j, ((label, _), rec, err) in enumerate(zip(entries, rec_np, errs)):
            print(f"  {name:6s} | {label:28s} | L2 rel {100 * _rel_l2(truth, rec):6.2f} %  | "
                  f"max |Δ|/||.||inf {err.max():5.2f} %")
            axs[row, 1 + j].imshow(rec, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
            im_e = axs[row, n + 1 + j].imshow(err, cmap="Reds", origin="lower",
                                              vmin=0.0, vmax=emax)

        # One colour bar per group: truth and reconstructions share a scale, and so do
        # the error maps, which is what makes the encoders comparable panel to panel.
        for col, im in ((n, im_f), (2 * n, im_e)):
            cb = fig.colorbar(im, cax=caxs[row][col])
            if col == 2 * n:
                cb.set_label(r"$|\Delta|\,/\,\|\cdot\|_\infty$ (%)")

    for j, (label, _) in enumerate(entries):
        axs[0, 1 + j].set_title(label, fontsize=MAP_TITLE_SIZE)
        axs[0, n + 1 + j].set_title(label, fontsize=MAP_TITLE_SIZE)
    axs[0, 0].set_title("Solver", fontsize=MAP_TITLE_SIZE)
    for row in (0, 1):
        for col in range(2 * n + 1):
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            if col not in (n, 2 * n):
                caxs[row][col].set_axis_off()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = save_figure(os.path.join(output_dir, f"recon_compare_{split}_snap{disp}.pdf"))
    plt.close()
    print(f"-> Figure saved to: {save_path}")
    return save_path


def reconstruction_report(encoder_kind, name, dataset=DEFAULT_DATASET, split="test",
                          field="both", snapshots=None, aggregate=False, grid=150,
                          output_dir=None, device="cpu", prefer_ft=False, splits=None,
                          encoder=None):
    """Full driver for the reconstruction quality of an encoder.

    Loads the `encoder_kind` encoder of experiment `name`, restricts the fields to the
    requested `split`, then:
      - `aggregate=True`: mean relative L2 over the WHOLE split -- the only mode that
        is comparable from one encoder to another;
      - `snapshots` not None: one figure per snapshot, from the same dataset load. It
        takes an int, an iterable of indices WITHIN the split, or a selection spec
        ("chaotic:1,random:2", cf. `resolve_snapshots`).

    The two are independent: `aggregate=True` with `snapshots="chaotic:1,random:2"`
    gives the split metric AND its figures in one pass.

    `prefer_ft=False` by default: we look at the encoder as trained, not at its version
    fine-tuned with the DeepONet (unlike `eval_one`, which must pair weights consistent
    with the model being evaluated). `encoder` overrides the whole loading step -- the
    way to judge a Transformer-Based DeepONet encoder as fine-tuned, since those weights
    live in the operator checkpoint and not under an `_ft` name.

    Returns the list of error dicts in aggregate mode, `None` otherwise.

        utils.reconstruction_report("pca", "PCA-p128")                      # metric only
        utils.reconstruction_report("tbd", "TBD-N4-D128", aggregate=True,
                                    snapshots="chaotic:1,random:2", split="test")
    """
    if snapshots is None and not aggregate:
        aggregate = True
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device) if isinstance(device, str) else device
    print(f"=== Reconstruction ({encoder_kind}) on: {device.type.upper()} ===")

    if encoder is None:
        encoder = load_any_encoder({"name": name, "encoder": encoder_kind}, device,
                                   prefer_ft=prefer_ft)
    is_joint = encoder_kind in ("tbd", "joint_ae")
    print(f"Encoder loaded ({encoder_kind}/{name}) | latent_dims = {encoder.latent_dims}")

    print(f"Loading dataset: {dataset} (split={split})")
    if split == "all":
        dict_data = torch.load(dataset, map_location="cpu", weights_only=False)["dict_data"]
        raw_indices = None
    else:
        splits = splits or load_and_split_dataset(dataset, map_location="cpu")
        dict_data = split_data(splits, split)
        raw_indices = splits[f"{split}_idx"]

    n_split = dict_data["u1_in"].shape[0]
    fields = ["u1_in", "u2_in"] if field == "both" else [field]

    stats = None
    if aggregate:
        print(f"\n=== Reconstruction error over the WHOLE '{split}' split "
              f"({n_split} snapshots) ===")
        stats = (aggregate_split_error_joint(encoder, dict_data, split=split) if is_joint
                 else [aggregate_split_error(encoder, dict_data, f, encoder_kind, split=split)
                       for f in fields])
    if snapshots is None:
        return stats

    if isinstance(snapshots, str):
        snap_list = resolve_snapshots(snapshots, dict_data)
    else:
        snap_list = [snapshots] if isinstance(snapshots, int) else [int(s) for s in snapshots]
    statut = {"val": "UNSEEN (validation)", "train": "SEEN (training)",
              "test": "UNSEEN (test)", "all": "unknown status (full dataset)"}[split]
    grid_shape = (grid, grid)

    for snap in snap_list:
        if not 0 <= snap < n_split:
            raise IndexError(
                f"snapshot {snap} out of range: the '{split}' split "
                f"contient {n_split} snapshots (indices 0..{n_split - 1})."
            )
        raw_idx = snap if raw_indices is None else int(raw_indices[snap])
        print(f"Local snapshot #{snap} of the '{split}' split "
              f"-> snapshot brut #{raw_idx} | {statut}")

        if is_joint:
            reconstruct_and_plot_joint(
                encoder, dict_data, snap, grid_shape=grid_shape,
                output_dir=output_dir, raw_idx=raw_idx, split=split)
        else:
            for field_key in fields:
                reconstruct_and_plot(
                    encoder, dict_data, field_key, snap, encoder_kind,
                    grid_shape=grid_shape, output_dir=output_dir,
                    raw_idx=raw_idx, split=split)
    return stats


def load_any_encoder(cfg, device, prefer_ft=True, save_dir=CHECKPOINTS_DIR):
    """Load the encoder of a model from cfg['encoder'] (+ cfg['name']).

    Prefers the fine-tuned encoder (_ft) when it exists: a checkpoint produced by a
    --finetune_encoder run evolved the AE jointly with the DeepONet, so evaluating with
    the original (frozen) encoder would pair inconsistent weights. Matches the same
    preference applied in run_eval_2d. `prefer_ft=False` forces the encoder as trained
    (for reconstruction figures: we judge the encoder, not the encoder+DeepONet pair)."""
    name, kind = cfg["name"], cfg["encoder"]
    if kind in ("joint_ae", "tbd"):
        from deeponet_joint import joint_encoder_path
        tag = "tbd" if kind == "tbd" else "ae"
        path, hint = joint_encoder_path(name, tag, save_dir=save_dir), (
            "deeponet_joint.py --mode train_encoder"
            + (" --joint_encoder tbd" if kind == "tbd" else ""))
        if kind == "tbd":
            from tbdeeponet import TBDEncoder
            loader = TBDEncoder.load
        else:
            from deeponet_joint import JointAutoencoderEncoder
            loader = JointAutoencoderEncoder.load
    else:
        from encoders import encoder_path, load_encoder
        ft_path = encoder_path(f"{name}_ft", kind, save_dir=save_dir)
        path = ft_path if (prefer_ft and os.path.exists(ft_path)) \
            else encoder_path(name, kind, save_dir=save_dir)
        hint = f"deeponet.py --mode train_encoder --encoder {kind}"
        loader = lambda p, device: load_encoder(kind, p, device=device)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Encodeur introuvable : {path}\nLancez d'abord : python {hint} --name {name}")
    return loader(path, device=device)


def load_any_model(cfg, device, save_dir=CHECKPOINTS_DIR):
    """Load the model from cfg['arch'] (modular | single)."""
    arch = cfg.get("arch", "modular")
    path = os.path.join(save_dir, f"{cfg['name']}.pth")
    if arch == "v2":
        from deeponet_v2 import load_model_v2
        return load_model_v2(path, device)
    if arch == "single":
        from deeponet_joint import load_model_single
        return load_model_single(path, device)
    from deeponet import load_model
    return load_model(path, device)


def eval_one(cfg, device, splits, with_residual=True, with_recon=False,
             batch_size_data=256, res_n_cells=4000, res_n_dirs=16, res_seed=0,
             which="test"):
    """Evaluate a model (any architecture) on the `which` split (TEST by default, cf.
    `split_data`) and return a dict of metrics: I-L2/L1, G-L2/L1, optionally the
    encoder reconstruction of T/kappa, optionally the residual rho/corr and
    RMS(r)/max(I). Keys that were not computed are None.

    Example:
        eval_one({"name":"PCA-p128-softplus","encoder":"pca","arch":"modular"},
                 device, splits, with_residual=True)  -> I_l2 on the test split
    """
    from deeponet import INPUT_KEYS
    from torch.utils.data import DataLoader, TensorDataset

    name = cfg["name"]
    print(f"\n{'#'*72}\n# {name}  ({cfg.get('label', name)}, "
          f"{cfg.get('arch', 'modular')}/{cfg['encoder']})\n{'#'*72}")
    encoder = load_any_encoder(cfg, device)
    model = load_any_model(cfg, device)

    data_eval = split_data(splits, which)
    y_full = splits["y_full"]

    evaluator = Evaluator(model, device)
    evaluator.encoder = encoder
    enc_eval = encoder.transform_dicts(
        {"u1_in": data_eval["u1_in"], "u2_in": data_eval["u2_in"]}, INPUT_KEYS)
    eval_loader = DataLoader(
        TensorDataset(enc_eval["u1_in"], enc_eval["u2_in"], data_eval["I_target"]),
        batch_size=batch_size_data, shuffle=False)
    val = evaluator.evaluate_split_errors(eval_loader, y_coords=y_full,
                                          label=which.upper())

    out = {"name": name, "label": cfg.get("label", name),
           "arch": cfg.get("arch", "modular"), "encoder": cfg["encoder"],
           "I_l2": val["I_l2"], "I_l1": val["I_l1"],
           "G_l2": val["G_l2"], "G_l1": val["G_l1"],
           "recon_T": None, "recon_k": None,
           "rho": None, "corr": None, "res_imax": None, "loss_imax": None}

    if with_recon:
        out["recon_T"] = aggregate_split_error(encoder, data_eval, "u1_in", cfg["encoder"], split=which)["mean"]
        out["recon_k"] = aggregate_split_error(encoder, data_eval, "u2_in", cfg["encoder"], split=which)["mean"]

    if with_residual:
        i_max = float(data_eval["I_target"].max())
        y, cell_idx, T, kappa = build_collocation(
            data_eval, y_full, res_n_dirs, res_n_cells,
            trained_dirs=False, seed=res_seed, device=device)
        acc = evaluate(model, encoder, y, cell_idx, T, kappa, device, seed=res_seed)
        rho, corr = report(acc, f"{name}_{which}", i_max=i_max)
        rms_r = math.sqrt(acc["r2"] / acc["n"])
        out["rho"], out["corr"] = rho, corr
        out["res_imax"] = rms_r / i_max
        out["loss_imax"] = (rms_r / i_max) ** 2

    return out


def plot_pca_modes(enc, n=5, grid=150, unscale=False, amplitude=False,
                   out="results/figures/pca_modes.pdf", enc_name=""):
    """Plot the first n eigenmodes (POD) of a PCA encoder, one field per row.
    --amplitude: x sqrt(var) (x scale_ with StandardScaler) -> physical field units."""
    labels = {"u1_in": r"$u_1 = T$  (temperature, K)",
              "u2_in": r"$u_2 = \kappa$  (absorption, m$^{-1}$)"}
    keys = list(enc.pcas.keys())
    g = grid
    fig, axs = plt.subplots(len(keys), n, figsize=(2.7 * n, 2.9 * len(keys)), squeeze=False)
    for row, key in enumerate(keys):
        pca = enc.pcas[key]
        scaler = enc.scalers.get(key)
        evr = pca.explained_variance_ratio_
        comps = pca.components_
        for col in range(n):
            mode = comps[col].copy()
            if amplitude:
                mode = np.sqrt(pca.explained_variance_[col]) * mode
                if scaler is not None:
                    mode = mode * scaler.scale_
            elif unscale and scaler is not None:
                mode = mode * scaler.scale_
            img = mode.reshape(g, g)
            vmax = np.abs(img).max()
            ax = axs[row, col]
            im = ax.imshow(img, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
            ax.set_title(f"mode {col + 1}\n{evr[col] * 100:.1f}% var", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axs[row, 0].set_ylabel(labels.get(key, key), fontsize=11)
    cum = enc.pcas[keys[0]].explained_variance_ratio_[:n].sum() * 100
    scal = ("StandardScaler (correlation)" if (enc.scalers.get(keys[0]) is not None)
            else "raw POD (covariance)")
    if amplitude:
        scal += " -- PHYSICAL AMPLITUDE (x sqrt(var), field units)"
    elif unscale:
        scal += " -- standardisation undone (unit eigenvector)"
    fig.suptitle(f"{enc_name}  |  {scal}  |  first {n} modes account for "
                 f"{cum:.1f}% of the variance ({keys[0]})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out = save_figure(out, fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out}")


def plot_pod_decomposition(enc, dataset=DEFAULT_DATASET, key="u1_in",
                           snapshot=0, n=8, grid=150, cmap="RdBu_r",
                           out="results/figures/pod_decomposition.pdf"):
    """POD decomposition of ONE snapshot: the anomaly x - x_bar plus the a_i*v_i
    contributions of the n modes, all on a common scale (same physical units)."""
    pca = enc.pcas[key]
    scaler = enc.scalers.get(key)
    V = pca.components_
    evr = pca.explained_variance_ratio_
    d = torch.load(dataset, map_location="cpu", weights_only=False)
    x = d["dict_data"][key][snapshot].numpy()
    z = scaler.transform(x[None])[0] if scaler is not None else x
    a = (z - pca.mean_) @ V.T
    s = scaler.scale_ if scaler is not None else 1.0

    def contrib(i):
        return a[i] * (s * V[i])

    mean_phys = (scaler.inverse_transform(pca.mean_[None])[0]
                 if scaler is not None else pca.mean_)
    dx = x - mean_phys
    partial = sum(contrib(i) for i in range(n))
    expl = 100 * (1 - np.linalg.norm(dx - partial) / np.linalg.norm(dx))
    unit = "K" if key == "u1_in" else r"m$^{-1}$"
    fname = {"u1_in": "T", "u2_in": r"\kappa"}.get(key, key)
    g = grid
    panels = [(rf"exemple : $x-\bar x$" + f"\n(snapshot {snapshot})", dx)]
    for i in range(n):
        panels.append((rf"mode {i+1} : $a_{{{i+1}}}v_{{{i+1}}}$" +
                       f"\n({evr[i]*100:.1f}% var)", contrib(i)))
    vmax = float(np.abs(dx).max())
    ncols = int(np.ceil(np.sqrt(len(panels))))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.9 * nrows))
    axs = np.atleast_1d(axs).ravel()
    im0 = None
    for k, (ax, (t, img)) in enumerate(zip(axs, panels)):
        im = ax.imshow(img.reshape(g, g), cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        im0 = im0 or im
        ax.set_title(t, fontsize=8.5, fontweight=("bold" if k == 0 else "normal"))
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(panels), len(axs)):
        axs[j].axis("off")
    cb = fig.colorbar(im0, ax=axs.tolist(), fraction=0.025, pad=0.02)
    cb.set_label(unit, fontsize=11)
    fig.suptitle(rf"POD decomposition of ${fname}$ -- common scale  |  "
                 f"snapshot {snapshot}  |  {n} modes ~ {expl:.0f}% of the anomaly", fontsize=12)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out = save_figure(out, fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out}  (n modes explain {expl:.1f}% of the anomaly)")


def plot_pod_modes_scaled(enc, key="u1_in", n=11, modes=None, grid=150,
                          mode_cmap="RdBu_r", mean_cmap="magma",
                          out="results/figures/pod_modes_scaled.pdf"):
    """POD board: mean field + modes weighted by sqrt(lambda) (typical physical
    amplitude), on a common scale with ONE colorbar for the modes. `modes` = 1-indexed
    indices (a list or 'a,b,c'), taking priority over n."""
    pca = enc.pcas[key]
    scaler = enc.scalers.get(key)
    V = pca.components_
    lam = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    s = scaler.scale_ if scaler is not None else 1.0
    mean = (scaler.inverse_transform(pca.mean_[None])[0] if scaler is not None else pca.mean_)
    g = grid
    unit = "K" if key == "u1_in" else r"m$^{-1}$"
    flabel = {"u1_in": "T", "u2_in": r"\kappa"}.get(key, key)
    if modes:
        seq = modes.split(",") if isinstance(modes, str) else modes
        mode_idx = [int(x) - 1 for x in seq]
    else:
        mode_idx = list(range(n))
    amp = [np.sqrt(lam[i]) * (s * V[i]) for i in mode_idx]
    vmax = max(float(np.abs(a).max()) for a in amp)
    total = len(mode_idx) + 1
    ncols = int(np.ceil(np.sqrt(total)))
    nrows = int(np.ceil(total / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.7 * nrows))
    axs = np.atleast_1d(axs).ravel()
    im_mean = axs[0].imshow(mean.reshape(g, g), cmap=mean_cmap, origin="lower")
    axs[0].set_title(rf"moyenne $\bar {flabel}$", fontsize=10, fontweight="bold")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    fig.colorbar(im_mean, ax=axs[0], fraction=0.046, pad=0.04).set_label(unit, fontsize=9)
    im_mode = None
    for slot, i in enumerate(mode_idx):
        ax = axs[slot + 1]
        im = ax.imshow(amp[slot].reshape(g, g), cmap=mode_cmap, origin="lower", vmin=-vmax, vmax=vmax)
        im_mode = im_mode or im
        ax.set_title(f"mode {i+1}\n{evr[i]*100:.2f}% var", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(total, len(axs)):
        axs[j].axis("off")
    cb = fig.colorbar(im_mode, ax=axs[1:total].tolist(), fraction=0.02, pad=0.02)
    cb.set_label(f"mode amplitude  [{unit}]", fontsize=10)
    kind = "POD covariance" if scaler is None else "PCA correlation"
    which = ("modes " + ", ".join(str(i + 1) for i in mode_idx)) if modes \
        else f"first {len(mode_idx)} modes"
    fig.suptitle(rf"PCA of ${flabel}$ -- mean + {which}, weighted by "
                 rf"$\sqrt{{\lambda_i}}$ (typical amplitude, in {unit})  --  {kind}", fontsize=12)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out = save_figure(out, fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out}  (vmax modes = {vmax:.1f} {unit})")


def plot_kappa_eigenmodes(enc, key="u2_in", n=15, grid=150, cmap="gray",
                          colorbar=False, amplitude=False,
                          out="results/figures/kappa_eigenmodes.pdf", enc_name=""):
    """An "eigenfaces" board: mean field + n eigenmodes on a grid."""
    pca = enc.pcas[key]
    scaler = enc.scalers.get(key)
    g = grid
    mean = scaler.mean_.copy() if scaler is not None else pca.mean_.copy()
    comps = pca.components_
    evr = pca.explained_variance_ratio_
    sign = 1.0
    total = n + 1
    ncols = int(np.ceil(np.sqrt(total)))
    nrows = int(np.ceil(total / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.35 * nrows))
    axs = np.atleast_1d(axs).ravel()
    flabel = {"u2_in": r"\kappa", "u1_in": "T"}.get(key, key)
    lam = pca.explained_variance_

    def add_cb(ax, im):
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    im0 = axs[0].imshow((sign * mean).reshape(g, g), cmap=cmap, origin="lower")
    axs[0].set_title(rf"moyenne ${flabel}$", fontsize=10, fontweight="bold")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    add_cb(axs[0], im0)
    for i in range(n):
        m = sign * comps[i]
        if amplitude:
            m = np.sqrt(lam[i]) * m
            if scaler is not None:
                m = m * scaler.scale_
        m = m.reshape(g, g)
        v = np.abs(m).max()
        im = axs[i + 1].imshow(m, cmap=cmap, origin="lower", vmin=-v, vmax=v)
        axs[i + 1].set_title(f"mode {i + 1}  ({evr[i] * 100:.1f}%)", fontsize=8)
        axs[i + 1].set_xticks([]); axs[i + 1].set_yticks([])
        add_cb(axs[i + 1], im)
    for j in range(total, len(axs)):
        axs[j].axis("off")
    kind = "POD covariance" if scaler is None else "PCA correlation (StandardScaler)"
    fig.suptitle(f"Eigenmodes of ${flabel}$  --  {kind}  ({enc_name})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out = save_figure(out, fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out}")


def plot_kappa_reconstruction(enc, dataset=DEFAULT_DATASET, key="u2_in",
                              snapshot=0, ks=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 500),
                              grid=150, cmap="magma",
                              out="results/figures/kappa_reconstruction.pdf", enc_name=""):
    """Progressive POD reconstruction of a field: the mean plus the truncated sum of
    the first k modes, for increasing k (the flame gets sharper and sharper)."""
    pca = enc.pcas[key]
    scaler = enc.scalers.get(key)
    d = torch.load(dataset, map_location="cpu", weights_only=False)
    X = d["dict_data"][key][snapshot:snapshot + 1].numpy()
    Xs = scaler.transform(X) if scaler is not None else X
    mean = pca.mean_
    coeffs = (Xs - mean) @ pca.components_.T

    def reconstruct(k):
        rec = mean + coeffs[:, :k] @ pca.components_[:k]
        if scaler is not None:
            rec = scaler.inverse_transform(rec)
        return rec[0]

    sign = 1.0
    g = grid
    ks = [int(s) for s in (ks.split(",") if isinstance(ks, str) else ks)]
    truth = sign * X[0]
    vmin, vmax = truth.min(), truth.max()
    panels = [("truth", truth)]
    cum = np.cumsum(pca.explained_variance_ratio_)
    for k in ks:
        rec = sign * reconstruct(k)
        err = np.linalg.norm(rec - truth) / np.linalg.norm(truth) * 100
        var = 0.0 if k == 0 else cum[min(k, len(cum)) - 1] * 100
        panels.append((f"{k} modes\n{var:.0f}% var | L2 {err:.1f}%", rec))
    ncols = int(np.ceil(np.sqrt(len(panels))))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.7 * nrows))
    axs = np.atleast_1d(axs).ravel()
    for ax, (title, img) in zip(axs, panels):
        ax.imshow(img.reshape(g, g), cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(panels), len(axs)):
        axs[j].axis("off")
    fig.suptitle(f"Reconstruction POD progressive de $\\kappa$ "
                 f"(snapshot {snapshot}, {enc_name})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    out = save_figure(out, fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {out}")


SNAPSHOT_SPEC_DEFAULT = "structure:8"
SNAPSHOT_RANDOM_SEED = 0


def resolve_snapshots(spec, data, k_default=4, field="G", metric="var", verbose=True,
                      seed=SNAPSHOT_RANDOM_SEED):
    """Turn a snapshot specification into a list of indices into `data`.

    Accepts, comma separated (or as a Python list):
      * an integer n >= 0       -> [n]
      * -1                      -> the MOST INTENSE snapshot (max(I_target))
      * "structure" / "structure:8" -> the k MOST STRUCTURED snapshots (largest
        structure score of G, cf. `snapshot_variance`): the long meandering plumes
        and detached vortices, the ones that actually read on a figure. The CLI
        default, and the selection used for the paper figures.
      * "chaotic" / "chaotic:4" -> the k MOST CHAOTIC snapshots (largest spatial
        variance of G, cf. `most_chaotic_snapshots`); k defaults to `k_default`.
        Beware: variance ranks by INTENSITY, so it returns compact bright blobs
        that all look alike -- prefer "structure" to pick a figure.
      * "random" / "random:4"   -> k snapshots DRAWN AT RANDOM among those not already
        selected -- they give an UNBIASED view of the model, where the chaotic ones
        alone show its worst case. The draw is SEEDED (`seed`, fixed by default), so
        every model is compared on the SAME random snapshots, evaluation after
        evaluation.
      * e.g. "structure:8" (the CLI default) or "structure:4,random:2,0,-1"

    Duplicates are removed, keeping the order of first appearance.
    """
    if isinstance(spec, (int, np.integer)):
        spec = [spec]
    elif isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    n_snap = int(data["I_target"].shape[0])
    out = []
    for s in spec:
        low = s.lower() if isinstance(s, str) else ""
        if low.startswith(("chaotic", "structure")):
            k = int(s.split(":")[1]) if ":" in s else k_default
            m = "structure" if low.startswith("structure") else metric
            idx, _ = most_chaotic_snapshots(data, k=k, field=field, metric=m,
                                            verbose=verbose)
            out += [int(i) for i in idx]
            continue
        if low.startswith(("random", "rand")):
            k = int(s.split(":")[1]) if ":" in s else k_default
            pool = np.setdiff1d(np.arange(n_snap), np.array(out, dtype=int))
            k = min(k, pool.size)
            pick = np.random.default_rng(seed).choice(pool, size=k, replace=False)
            pick = np.sort(pick)
            out += [int(i) for i in pick]
            if verbose:
                print(f"  {k} snapshots drawn at random (seed={seed}, hence identical "
                      f"from one model to the next): {[int(i) for i in pick]}")
            continue
        i = int(s)
        if i < 0:
            i = int(data["I_target"].max(dim=1).values.argmax())
            if verbose:
                print(f"  most intense snapshot: {i}")
        out.append(i)
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i); uniq.append(i)
    return uniq


def plot_eval_snapshots(evaluator, dict_data, y_full, spec=SNAPSHOT_SPEC_DEFAULT,
                        output_dir="results/evaluation", include_unseen_angles=False,
                        err_mode="peak", field="G"):
    """G truth / G model / relative error % maps for a SELECTION of snapshots, then a
    summary of the per-snapshot errors.

    The common entry point of the three evaluation modes (deeponet eval_2d,
    deeponet_joint eval_2d and eval_operator), so they share exactly the same snapshot
    selection, the same figures and the same summary.

    `spec`: cf. `resolve_snapshots` -- by default 4 chaotic snapshots (the model's
    worst case) plus 4 drawn at a fixed seed (an unbiased view), i.e. 8 figures.

    Returns the list of dicts returned by `plot_snapshot_comparison_dataset`.
    """
    snaps = resolve_snapshots(spec, dict_data, field=field)
    if not snaps:
        print("no snapshot selected -> no figure.")
        return []
    n_dirs = evaluator._infer_direction_count(y_full)
    var = snapshot_variance(dict_data, field=field, n_dirs=n_dirs)
    rank = np.argsort(np.argsort(-var)) + 1
    print(f"snapshots plotted ({len(snaps)}): {snaps}")

    # One G scale for the whole set: the peak of the brightest reference plotted. The
    # figures of two snapshots (and a fortiori of two models) then share every colour.
    I_sel = dict_data["I_target"][list(snaps)].float()
    g_vmax = float((I_sel.view(len(snaps), -1, n_dirs).sum(dim=2)
                    * dir_weight(n_dirs)).max())
    print(f"shared colour scales: G in [0, {g_vmax:.3e}], "
          f"error in [0, {float(ERR_VMAX or ERR_VMAX_DEFAULT):.0f} %]")

    rows = []
    for snap in snaps:
        rows.append(evaluator.plot_snapshot_comparison_dataset(
            dict_data, snapshot_idx=snap, y_full=y_full, output_dir=output_dir,
            include_unseen_angles=include_unseen_angles, err_mode=err_mode,
            g_vmax=g_vmax))

    print(f"\n--- Relative G-L2 error per plotted snapshot "
          f"(variance rank among the {len(var)} snapshots of the split) ---")
    for snap, r in zip(snaps, rows):
        print(f"  snapshot {snap:4d}   variance rank {rank[snap]:4d}/{len(var)}"
              f"   G-L2 = {r['pred16']*100:6.2f}%")
    errs = np.array([r["pred16"] for r in rows])
    print(f"  -> mean over these {len(errs)} snapshots: {100*errs.mean():.2f}% "
          f"(min {100*errs.min():.2f}%, max {100*errs.max():.2f}%)\n")
    return rows


def plot_pred_compare(models=("expPCA100", "expPCA10", "expPCA3"), snapshot=-1,
                      dataset=DEFAULT_DATASET,
                      out="results/figures/pred_compare.pdf", device=None,
                      err_mode="peak", with_error_row=True, splits=None, which="test"):
    """Compare G = integral of I over phi as predicted by several models against the
    truth, on a common scale, with each model's RELATIVE ERROR MAP in % underneath.

      row 1: G truth | G model 1 | G model 2 | ...  (scale set by the truth)
      row 2:   --    | error %   | error %   | ...  (scale shared by all)

    `snapshot`: cf. `resolve_snapshots` -- an integer, -1 (the most intense), a list, or
    "chaotic[:k]" for the k snapshots of largest spatial variance (the most structured,
    hence the most discriminating between models). When several snapshots are asked
    for, one figure is written per snapshot, suffixed `_snap<idx>`.

    Uses ONLY the .pt dataset plus the mesh (mesh_stub.h5 as a fallback): unlike
    `Evaluator.plot_snapshot_comparison` it needs no time_groups from the real H5, so it
    works for any architecture/encoder already present in the .pt.

    `models`: a list of names (str, back-compat = pca/modular) OR of cfg dicts
    {"name":.., "encoder": "pca"|"autoencoder"|"joint_ae"|"tbd"|
    "tbd_operator", "arch": "modular"|"single", "label":..}.
    "tbd_operator" = a full TBD->I operator (encoder+decoder in the same checkpoint
    checkpoints/operator_tbd_<name>.pt, cf. tbdeeponet.load_operator); it ignores "arch" and
    loads encoder and model together.

    Returns {snapshot_idx: {label: I-L2}}."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(models, str):
        models = models.split(",")
    cfgs = [{"name": m, "encoder": "pca", "arch": "modular"} if isinstance(m, str) else dict(m)
            for m in models]

    h5 = solver_h5()

    if splits is None:
        splits = load_and_split_dataset(dataset, map_location="cpu")
    data_val = split_data(splits, which)
    y_full = splits["y_full"]
    indices = resolve_snapshots(snapshot, data_val)
    print(f"snapshots {which} choisis : {indices}")

    loaded = []
    for cfg in cfgs:
        if cfg.get("encoder") == "tbd_operator":
            from tbdeeponet import load_operator
            from deeponet_joint import operator_path
            model, enc = load_operator(operator_path(cfg["name"]), device=device)
        else:
            model = load_any_model(cfg, device)
            enc = load_any_encoder(cfg, device)
        ev = Evaluator(model, device, h5_path=h5); ev.encoder = enc
        loaded.append((cfg, ev))
    n_dirs = loaded[0][1]._infer_direction_count(y_full)
    loaded[0][1].ensure_grid(y_full)
    X, Y = loaded[0][1].X_grid, loaded[0][1].Y_grid
    var_all = snapshot_variance(data_val, field="G", n_dirs=n_dirs)
    rank_all = np.argsort(np.argsort(-var_all)) + 1

    all_errs = {}
    for idx in indices:
        u1 = data_val["u1_in"][idx:idx + 1]
        u2 = data_val["u2_in"][idx:idx + 1]
        I_true = data_val["I_target"][idx:idx + 1].float()
        G_preds, errs = {}, {}
        for cfg, ev in loaded:
            u1b, u2b = ev._encode_inputs(u1, u2)
            I_pred = ev._predict_chunked([u1b, u2b], y_full)
            G_preds[cfg["name"]] = ev._integrate_over_directions(
                I_pred.cpu(), n_dirs).squeeze().numpy()
            l2, _ = ev.calculate_relative_norms(I_pred.cpu(), I_true)
            errs[cfg["name"]] = float(l2)
        G_true = loaded[0][1]._integrate_over_directions(I_true, n_dirs).squeeze().numpy()

        lv = np.linspace(float(G_true.min()), float(G_true.max()), 40)
        err_maps = {n: rel_error_map(G, G_true, mode=err_mode) for n, G in G_preds.items()}
        e_lv, _ = sym_error_levels(list(err_maps.values()))

        ncols = 1 + len(cfgs)
        nrows = 2 if with_error_row else 1
        fig, axs = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.4 * nrows),
                                squeeze=False)
        im = axs[0, 0].contourf(X, Y, G_true.reshape(150, 150), levels=lv,
                                cmap="jet", extend="both")
        axs[0, 0].set_title("TRUTH", fontsize=11)
        for j, cfg in enumerate(cfgs, start=1):
            im = axs[0, j].contourf(X, Y, G_preds[cfg["name"]].reshape(150, 150),
                                    levels=lv, cmap="jet", extend="both")
            axs[0, j].set_title(f"{cfg.get('label', cfg['name'])}\n"
                                f"I-L2 = {errs[cfg['name']]*100:.1f}%", fontsize=11)
        fig.colorbar(im, ax=axs[0, :].tolist(), fraction=0.025, pad=0.02, label="G")

        if with_error_row:
            axs[1, 0].axis("off")
            for j, cfg in enumerate(cfgs, start=1):
                e = err_maps[cfg["name"]]
                ime = axs[1, j].contourf(X, Y, e.reshape(150, 150), levels=e_lv,
                                         cmap=ERR_CMAP, extend="both")
                axs[1, j].set_title(_err_title(e, "signed relative error (% of the G peak)"),
                                    fontsize=10)
            fig.colorbar(ime, ax=axs[1, 1:].tolist(), fraction=0.025, pad=0.02,
                         format="%+.1f%%")

        for ax in axs.flat:
            if ax.has_data():
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"G prediction -- snapshot {idx} "
                     f"(spatial variance rank {rank_all[idx]}/{len(var_all)}, "
                     f"common scale)", fontsize=13)

        path = out if len(indices) == 1 else \
            f"{os.path.splitext(out)[0]}_snap{idx}{os.path.splitext(out)[1]}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        path = save_figure(path, fig, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure : {path}")
        for cfg in cfgs:
            print(f"  {cfg.get('label', cfg['name'])}: I-L2 = {errs[cfg['name']]*100:.2f}%")
        all_errs[idx] = {cfg.get("label", cfg["name"]): errs[cfg["name"]] for cfg in cfgs}
    return all_errs


DEFAULT_H5 = SOLVER_H5

# Default of `plot_val_error_vs_time`: the three ranks of the paper's PCA-DeepONet.
PCA_SWEEP = [{"name": f"PCA-p{p}", "encoder": "pca", "arch": "modular",
              "label": f"PCA-p{p}"} for p in (128, 256, 512)]


def snapshot_times(splits, which="test", h5_path=DEFAULT_H5):
    """Physical time (s) of every snapshot of the requested split, ALIGNED with the
    row order of splits['data'][...] (hence with splits[f'{which}_idx']).

    Rebuilds the EXACT order of prepare_data.py (`sorted(time_*)`) and reads the
    .attrs['time'] attribute of the H5. Returns (times, idx), where `idx` is the
    original snapshot index (= position in the .pt) of each row of the split.
    """
    with h5py.File(h5_path, "r") as f:
        # Alphabetical, matching prepare_data.py: time_10 sorts before time_9,
        # so .pt snapshot order is NOT chronological.
        keys = sorted(k for k in f.keys() if k.startswith("time_"))
        times_all = np.array([f[k].attrs["time"] for k in keys], dtype=np.float64)

    if which == "all":
        idx = np.arange(times_all.shape[0])
    else:
        idx = splits[f"{which}_idx"].numpy()

    n_split = (splits["data"][0]["u1_in"].shape[0]
               + splits["data"][1]["u1_in"].shape[0]
               + splits["data_test"]["u1_in"].shape[0])
    if times_all.shape[0] != n_split:
        print("  [!] the number of time_ groups in the H5 differs from the number of "
              "snapshots in the dataset: the time-to-snapshot mapping may be wrong.")
    return times_all[idx], idx


@torch.no_grad()
def per_snapshot_val_errors(cfg, device, splits, metric="I", batch_size_data=256,
                            which="test"):
    """PER-SNAPSHOT relative L2 error over the `which` split (TEST by default), in the
    order of splits[f'{which}_idx'] (so it pairs directly with
    snapshot_times(..., which)).

    metric: 'I' (raw directional intensity) or 'G' (integrated over the directions).
    Returns an np.ndarray of shape (n_snapshots of the split,).
    """
    from deeponet import INPUT_KEYS
    from torch.utils.data import DataLoader, TensorDataset

    encoder = load_any_encoder(cfg, device)
    model = load_any_model(cfg, device)
    model.eval()

    data_val = split_data(splits, which)
    y_full = splits["y_full"]

    ev = Evaluator(model, device)
    ev.encoder = encoder
    n_dirs = ev._infer_direction_count(y_full)

    enc_val = encoder.transform_dicts(
        {"u1_in": data_val["u1_in"], "u2_in": data_val["u2_in"]}, INPUT_KEYS)
    loader = DataLoader(
        TensorDataset(enc_val["u1_in"], enc_val["u2_in"], data_val["I_target"]),
        batch_size=batch_size_data, shuffle=False)

    errs = []
    for u1_in, u2_in, tgt in loader:
        u_in = [u1_in.to(device), u2_in.to(device)]
        tgt = tgt.to(device)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        pred = ev._predict_chunked(u_in, y_full)
        if metric == "G":
            pred = ev._integrate_over_directions(pred, n_dirs)
            tgt = ev._integrate_over_directions(tgt, n_dirs)
        l2, _ = ev._per_snapshot_norms(pred.cpu(), tgt.cpu())
        errs.append(l2.numpy())
    return np.concatenate(errs)


def plot_val_error_vs_time(cfgs=None, device=None, splits=None, metric="I",
                           dataset=DEFAULT_DATASET, h5_path=DEFAULT_H5,
                           out_dir="results/figures", prefix="val_error_vs_time",
                           n_time_bins=30, trend="mean"):
    """L2 errors ordered by real PHYSICAL TIME -- ONE FIGURE PER MODEL.

    For each model of `cfgs`, writes `{out_dir}/{prefix}_{name}.pdf`: a per-snapshot
    (time, error) scatter.

    trend: an aggregate curve per time window (n_time_bins equal-width windows)
    overlaid on the scatter, to spot a region the model learns badly:
      * "mean" (default) -- mean per window; a flat curve means no regime is
        systematically learned worse;
      * "p95"            -- 95th percentile per window; it spikes if a region
        CONCENTRATES the catastrophic snapshots (better for a localised failure);
      * None             -- scatter only.

    The time of each snapshot is read back from the H5 (cf. snapshot_times), because
    the .pt does not store it and the .pt index is not chronological.

    Returns {label: {'t': times, 'err': errs}} for reuse.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfgs is None:
        cfgs = PCA_SWEEP
    if isinstance(cfgs, dict):
        cfgs = [cfgs]
    if splits is None:
        splits = load_and_split_dataset(dataset, map_location="cpu")

    t_val, _ = snapshot_times(splits, "val", h5_path)
    order = np.argsort(t_val)
    t_sorted = t_val[order]
    edges = np.linspace(t_sorted.min(), t_sorted.max(), n_time_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which_bin = np.clip(np.digitize(t_sorted, edges) - 1, 0, n_time_bins - 1)

    os.makedirs(out_dir, exist_ok=True)
    results = {}
    print(f"\n=== VALIDATION {metric}-L2 ERROR vs PHYSICAL TIME "
          f"({len(t_val)} snapshots, t∈[{t_sorted.min():.1f}, {t_sorted.max():.1f}] s) ===")
    for cfg in cfgs:
        label = cfg.get("label", cfg["name"])
        err = per_snapshot_val_errors(cfg, device, splits, metric=metric)
        results[label] = {"t": t_val, "err": err}
        e_sorted = err[order]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(t_sorted, 100 * e_sorted, s=10, alpha=0.35, color="steelblue",
                   label="snapshot val")
        if trend is not None:
            agg = (lambda v: np.percentile(v, 95)) if trend == "p95" else np.mean
            binned = np.array([agg(e_sorted[which_bin == b]) if np.any(which_bin == b)
                               else np.nan for b in range(n_time_bins)])
            ax.plot(centers, 100 * binned, "-", lw=2.2, color="crimson",
                    label=f"{trend} / window")
        ax.set_xlabel("physical time (s)")
        ax.set_ylabel(f"per-snapshot relative {metric}-L2 error (%)")
        ax.set_title(f"{label} -- {metric}-L2 error ordered by time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        out = os.path.join(out_dir, f"{prefix}_{cfg['name']}.pdf")
        out = save_figure(out, fig, bbox_inches="tight")
        plt.close(fig)

        r = np.corrcoef(t_val, err)[0, 1]
        print(f"  {label:<10} {metric}-L2 mean={100*err.mean():5.2f}%  med={100*np.median(err):5.2f}%"
              f"  max={100*err.max():6.2f}%  corr(err,t)={r:+.2f}   -> {out}")

    print()
    return results


def save_atomic(obj, path):
    """A torch.save that cannot leave a half-written file behind."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def resume_path(name, kind="train", save_dir=CHECKPOINTS_DIR):
    """checkpoints/<name>_<kind>_resume.pt -- kind separates encoder from operator."""
    return os.path.join(save_dir, f"{name}_{kind}_resume.pt")


def rel_phys_loss(residual, I, u1_eval, u2_eval, return_parts=False):
    """RELATIVE physics loss (= squared relative residual), the project's single
    normalisation.

    Denominator = max(RMS(transport)^2, RMS(rhs)^2), detached: the residual has the
    units of a kappa*I (not of an I), so normalising by I_scale^2 gave values around
    1e2-1e3 regardless of model quality. Here we get an O(1) number comparable with the
    data loss, and the .detach() stops the optimiser from cheating by inflating the
    transport term.

    residual, I, u1_eval, u2_eval: (B, P). Returns a torch scalar.

    return_parts=True also returns mean(r^2) -- the residual MSE in physical units,
    normalised by the number of points ONLY -- and the denominator, so a run can log
    the ABSOLUTE physics curve, which the ratio alone does not allow to reconstruct
    afterwards (the denominator is re-measured on every batch).
    """
    rhs = u1_eval + u2_eval * I
    transport = residual + rhs
    denom = torch.maximum((transport ** 2).mean(),
                          (rhs ** 2).mean()).detach().clamp_min(1e-12)
    mse = torch.mean(residual ** 2)
    if return_parts:
        return mse / denom, mse.detach(), denom
    return mse / denom


class TrainingHistory:
    """Loss history of a training run: {series -> (epochs, values)}.

    Each series carries ITS OWN epochs: the data loss is recorded every epoch, the
    physics loss only at the monitoring cadence, the validation only at validated
    epochs -- a single aligned array would be wrong or full of holes.

    Naming convention: "<split>_<term>" with split in {train, val} and term in
    {data, phys, bc, total}. `plot_learning_curves` groups the panels by term.

    Saved as JSON next to the checkpoint (`checkpoints/<name>_history.json`): light,
    readable without torch, and enough to REPLOT or compare runs after the fact
    without retraining.
    """

    def __init__(self, name, save_dir=CHECKPOINTS_DIR, meta=None):
        self.name = name
        self.save_dir = save_dir
        self.meta = dict(meta or {})
        self.series = {}

    def log(self, epoch, **values):
        """log(12, train_data=0.4, val_data=0.5, train_phys=0.39) -- None values are
        ignored, so the same keywords can be passed whether a loss was computed or not."""
        for key, val in values.items():
            if val is None:
                continue
            val = float(val)
            if not math.isfinite(val):
                continue
            s = self.series.setdefault(key, {"epoch": [], "value": []})
            s["epoch"].append(int(epoch))
            s["value"].append(val)

    @staticmethod
    def path(name, save_dir=CHECKPOINTS_DIR):
        return os.path.join(save_dir, f"{name}_history.json")

    def save(self, verbose=True):
        import json
        os.makedirs(self.save_dir, exist_ok=True)
        p = self.path(self.name, self.save_dir)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "meta": self.meta, "series": self.series}, f)
        if verbose:
            print(f"Loss history saved to: {p}")
        return p

    @classmethod
    def load(cls, name, save_dir=CHECKPOINTS_DIR):
        import json
        with open(cls.path(name, save_dir), encoding="utf-8") as f:
            d = json.load(f)
        h = cls(d["name"], save_dir=save_dir, meta=d.get("meta", {}))
        h.series = d["series"]
        return h


_DEFAULT_CURVE_TERMS = ("data", "phys", "phys_abs", "bc", "total")


def plot_learning_curves(names, save_dir=CHECKPOINTS_DIR, output=None, title=None,
                         terms=_DEFAULT_CURVE_TERMS, logy=True):
    """Plot the learning curves of one or SEVERAL runs from their JSON history.

    names  : a run name, a list of names (compared on the same axes), or a list of
             (name, label) pairs.
    output : path of the figure (default: results/figures/<name>_learning_curves.pdf,
             or
             _compare_learning_curves.pdf for several runs); the extension is set
             by `save_figure`.
    terms  : terms plotted when present -- one panel per term actually found. The
             physics loss shows up even for a data-only run if monitoring was on.
             "phys_abs" (mean(r^2), physical units) replaces the relative "phys"
             panel when the run recorded it; older runs only have the relative one.
             That substitution applies to the DEFAULT terms only -- pass both
             explicitly to get both panels, which is the way to see that they are the
             same curve up to the (model-dependent) denominator.

    Train as a solid line, val dashed with markers; one colour per run.
    Usable AFTER THE FACT on any run that has its <name>_history.json.
    """
    if isinstance(names, str):
        names = [names]
    runs = [(n, n) if isinstance(n, str) else tuple(n) for n in names]

    hists = []
    for name, label in runs:
        try:
            hists.append((label, TrainingHistory.load(name, save_dir)))
        except FileNotFoundError:
            print(f"  [!] no history for '{name}' "
                  f"({TrainingHistory.path(name, save_dir)}) -- run older than the logging?")
    if not hists:
        raise FileNotFoundError("No history found for the requested runs.")

    present = [t for t in terms
               if any(f"{s}_{t}" in h.series for _, h in hists for s in ("train", "val"))]
    if "total" in present and "data" in present and all(
            h.series.get("train_total", {}).get("value") ==
            h.series.get("train_data", {}).get("value") for _, h in hists):
        present.remove("total")
    if terms is _DEFAULT_CURVE_TERMS and "phys_abs" in present and "phys" in present:
        present.remove("phys")
    if not present:
        raise ValueError("No usable series in the histories.")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 4.6), squeeze=False)
    for ax, term in zip(axes[0], present):
        for k, (label, h) in enumerate(hists):
            c = colors[k % len(colors)]
            for split, style in (("train", dict(ls="-", lw=1.6)),
                                 ("val", dict(ls="--", lw=1.1, marker="o", ms=3, alpha=0.8))):
                s = h.series.get(f"{split}_{term}")
                if not s:
                    continue
                ax.plot(s["epoch"], s["value"], color=c,
                        label=f"{label} -- {split}", **style)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel({"data": r"data loss",
                       "phys": r"relative RTE residual$^2$",
                       "phys_abs": r"RTE residual$^2$ per point",
                       "bc": r"MSE$(I_{bc})/I_{scale}^2$",
                       "total": "weighted total loss"}.get(term, term))
        ax.set_title(f"loss {term.upper()}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(title or (hists[0][0] if len(hists) == 1 else "comparaison de runs"),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    if output is None:
        if len(runs) == 1:
            out_dir, stem = os.path.join(RESULTS_DIR, runs[0][0]), runs[0][0]
        else:
            out_dir, stem = os.path.join(RESULTS_DIR, "comparisons"), "compare"
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"{stem}_learning_curves.pdf")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    output = save_figure(output, fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curves -> {output}")
    return output
