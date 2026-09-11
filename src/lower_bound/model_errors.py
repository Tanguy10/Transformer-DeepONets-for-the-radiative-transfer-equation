import fnmatch
import glob
import os
import sys

import numpy as np
import torch

_RTE2D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RTE2D not in sys.path:
    sys.path.insert(0, _RTE2D)


def default_ckpt_dir():
    """The repository's checkpoint directory, where every model of the paper lands."""
    from utils import CHECKPOINTS_DIR
    return CHECKPOINTS_DIR


def affine_head(name):
    """True for the softplus-free runs, whose reconstructor is LITERALLY the R of
    Theorem 3.4 (I = (<beta, gamma(y)> + b) * I_scale) where a softplus run is only
    affine up to that monotone map.

    They are named `<run>-affine` and reuse the encoder of `<run>`, exactly as their
    training did; the head itself is read back from the checkpoint by
    `deeponet.load_model`, which needs no help from the name."""
    return name.endswith("-affine")


def discover_models(ckpt_dirs):
    """The models sitting in `ckpt_dirs`, as (name, kind, p, affine, ckpt_dir) tuples.

    A model is recognised by its checkpoints: `operator_tbd_<name>.pt` is a
    Transformer-Based DeepONet, `<name>.pth` next to `encoder_<kind>_<name>.<ext>` is a
    DeepONet with that encoder. The width p is READ from the checkpoint -- p_basis of
    the branch.trunk junction, n_latent * dim for the attention decoder -- so dropping a
    new ablation in the directory is all it takes for it to enter the comparison.

    An AFFINE run (`<run>-affine`) carries no encoder of its own: it reuses the one of
    the softplus run it was derived from, so its family is read off the name instead
    (cf. `affine_head`)."""
    from encoders import _ENCODER_EXT, encoder_run
    from deeponet import infer_architecture

    out = []
    for ckpt_dir in ([ckpt_dirs] if isinstance(ckpt_dirs, str) else list(ckpt_dirs)):
        for path in sorted(glob.glob(os.path.join(ckpt_dir, "operator_tbd_*.pt"))):
            name = os.path.basename(path)[len("operator_tbd_"):-len(".pt")]
            arch = torch.load(path, map_location="cpu", weights_only=False)["arch"]
            out.append((name, "tbd", int(arch["n_latent"]) * int(arch["dim"]), False,
                        ckpt_dir))
        for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.pth"))):
            name = os.path.basename(path)[:-len(".pth")]
            kind = next((k for k, ext in _ENCODER_EXT.items()
                         if os.path.exists(os.path.join(
                             ckpt_dir, f"encoder_{k}_{encoder_run(name)}.{ext}"))), None)
            if kind is None:
                continue
            p = infer_architecture(torch.load(path, map_location="cpu"))[2]
            out.append((name, kind, int(p), True, ckpt_dir))
    rank = {"pca": 0, "autoencoder": 1, "tbd": 2}
    return sorted(out, key=lambda r: (rank.get(r[1], 9), r[2], r[0]))


def select_models(models, patterns):
    """The subset of `models` matched by the shell patterns of --models.

    Matching is case-insensitive and applies to the name with and without its family
    prefix, so both `--models 'TBD-*'` and `--models 'N4-*'` pick TBD-N4-D128. A pattern
    that matches nothing is an error: silently plotting fewer models than asked would be
    worse."""
    if not patterns:
        return models
    kept, unseen = [], []
    for pat in patterns:
        hit = [m for m in models
               if fnmatch.fnmatch(m[0].lower(), pat.lower())
               or fnmatch.fnmatch(m[0].split("-", 1)[-1].lower(), pat.lower())]
        if not hit:
            unseen.append(pat)
        kept += [m for m in hit if m not in kept]
    if unseen:
        raise SystemExit(f"--models: no match for {unseen}\n"
                         f"available: {', '.join(m[0] for m in models)}")
    return [m for m in models if m in kept]


def _h5():
    """The solver H5 if reachable, else mesh_stub: this does not affect the error (it
    y_coords), sert seulement au constructeur d'Evaluator.

    Goes through `utils.solver_h5`, which does NOT probe an unmounted automount: an
    `os.path.exists()` on that path while the host is down freezes the process in state D
    (unkillable, even by SIGKILL)."""
    from utils import solver_h5
    return solver_h5()


def split_data(splits, which):
    """The snapshot dict of the requested split ('train' | 'val' | 'test').

    Here `which` is ALWAYS explicit: this package is the one exception to the "report on
    test" rule -- Thm 3.4 bounds each dataset with ITS OWN spectrum, so
    E_hat (a non-relative error) is measured on train AND test."""
    from utils import split_data as _split_data
    return _split_data(splits, which)


@torch.no_grad()
def squared_errors(model, encoder, splits, device, which, batch_size=256):
    """||I_m - N(u_m)||^2 per snapshot of the `which` split ('train'|'val'|'test').

    float64, the ABSOLUTE error of Thm 3.4 -- the one the bound speaks about. The
    relative errors of Table 1 are not this package's business: `utils.eval_one`."""
    from deeponet import INPUT_KEYS
    from utils import Evaluator
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    data = split_data(splits, which)
    y_full = splits["y_full"]

    ev = Evaluator(model, device, h5_path=_h5())
    ev.encoder = encoder
    enc = encoder.transform_dicts(
        {"u1_in": data["u1_in"], "u2_in": data["u2_in"]}, INPUT_KEYS)
    loader = DataLoader(TensorDataset(enc["u1_in"], enc["u2_in"], data["I_target"]),
                        batch_size=batch_size, shuffle=False)

    out = []
    for u1_in, u2_in, tgt in loader:
        u_in = [u1_in.to(device), u2_in.to(device)]
        tgt = tgt.to(device)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        pred = ev._predict_chunked(u_in, y_full)
        out.append((pred.double() - tgt.double()).pow(2).sum(dim=1).cpu().numpy())
    return np.concatenate(out)


def _deeponet(name, enc_kind, splits, device, which, ckpt_dir=None):
    from utils import CHECKPOINTS_DIR, load_any_model, load_any_encoder
    ckpt_dir = ckpt_dir or CHECKPOINTS_DIR
    cfg = {"name": name, "encoder": enc_kind, "arch": "modular"}
    model = load_any_model(cfg, device, save_dir=ckpt_dir)
    encoder = load_any_encoder(cfg, device, save_dir=ckpt_dir)
    return squared_errors(model, encoder, splits, device, which)


def pca_deeponet(splits, device, which, name="PCA-deeponet", ckpt_dir=None):
    return _deeponet(name, "pca", splits, device, which, ckpt_dir)


def ae_deeponet(splits, device, which, name="AE-deeponet", ckpt_dir=None):
    return _deeponet(name, "autoencoder", splits, device, which, ckpt_dir)


def tbd_operator(splits, device, which, name="TBD-finetune", ckpt_dir=None, batch_size=32):
    """Reduced batch_size: the attention decoder keeps a (B x chunk x dim) activation
    per block -- at B = 256 the peak exceeds 5 GB and OOMs as soon as a training run is
    on the GPU. The DeepONets do not have this problem."""
    from tbdeeponet import load_operator
    from utils import CHECKPOINTS_DIR
    operator, encoder = load_operator(
        os.path.join(ckpt_dir or CHECKPOINTS_DIR, f"operator_tbd_{name}.pt"), device=device)
    return squared_errors(operator, encoder, splits, device, which, batch_size=batch_size)


def E_hat(sqerr):
    """E_hat = sqrt( (1/M) sum_m ||I_m - N(u_m)||^2 ), the left-hand side of Thm 3.4."""
    return float(np.sqrt(np.mean(sqerr)))
