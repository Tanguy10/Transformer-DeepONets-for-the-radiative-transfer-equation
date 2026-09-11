#!/usr/bin/env python
"""RTE 2D DeepONet with a single branch net fed by a joint autoencoder.

One conv autoencoder encodes both input fields (T, kappa) stacked as 2 channels
into a shared latent z, standardised per channel. Contract used by Trainer and
Evaluator: transform_dicts returns the SAME z under both the u1_in and u2_in
keys, and the single-branch model reads only the first.

I = softplus(branch(z) . trunk(y) + b) * I_scale. Trainer, collocation,
ReLoBRaLo and Evaluator are reused unchanged from deeponet / utils.
CLI examples at the bottom of this file.
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call, jacfwd, vmap
from torch.utils.data import DataLoader, TensorDataset

from deeponet import (
    MLP, Trainer, FullGridDataset, CollocationSampler, INPUT_KEYS,
)
from utils import (Evaluator, load_and_split_dataset, plot_eval_snapshots, transport_coeffs,
                   split_data, solver_h5,
                   SNAPSHOT_SPEC_DEFAULT, DEFAULT_DATASET, MESH_STUB, CHECKPOINTS_DIR, RESULTS_DIR)


class JointConvAutoencoder(nn.Module):
    """Same CNN AE as ConvAutoencoder but with 2 channels in AND out.

    (N, 2, 150, 150) --enc--> z (N, latent_dim) --dec--> (N, 2, 150, 150).
    15x15 bottleneck (150 /2 /5), like the single-channel autoencoders of the project.
    """

    def __init__(self, latent_dim=512, base=32, in_channels=2):
        super().__init__()
        self.base = base
        self.in_channels = in_channels

        def enc_block(ci, co, pool):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                nn.MaxPool2d(pool))

        self.encoder_conv = nn.Sequential(enc_block(in_channels, base, 2),
                                          enc_block(base, base * 2, 5))
        self.enc_fc = nn.Linear(base * 2 * 15 * 15, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, base * 2 * 15 * 15)

        def dec_block(ci, co, scale):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())

        self.decoder_conv = nn.Sequential(
            dec_block(base * 2, base, 5), dec_block(base, base, 2),
            nn.Conv2d(base, in_channels, 3, padding=1))

    def encode(self, x):
        """(N, 2, H, W) -> z (N, latent_dim)."""
        h = self.encoder_conv(x)
        return self.enc_fc(h.flatten(1))

    def forward(self, x):
        z = self.encode(x)
        out = self.dec_fc(z).view(-1, self.base * 2, 15, 15)
        return self.decoder_conv(out), z


class JointAutoencoderEncoder:
    """JOINT encoder: a single AE mapping (T, kappa) to one shared latent z.

    Compatible with Trainer/Evaluator: transform_dicts returns z under BOTH the
    u1_in and u2_in keys (SingleBranchDeepONet only reads branch_inputs[0]).
    """

    kind = "joint_autoencoder"

    def __init__(self, latent_dim=512, grid_shape=(150, 150), base=32,
                 epochs=600, batch_size=64, lr=1e-3, device=None, encode_batch=256):
        self.latent_dim = latent_dim
        self.grid_shape = tuple(grid_shape)
        self.base = base
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encode_batch = encode_batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.stats = None               
        self.latent_dims = {"u1_in": latent_dim, "u2_in": latent_dim}
        self.fitted = False

    def _stack(self, u1, u2):
        """(N, N_cells), (N, N_cells) -> (N, 2, H, W)."""
        H, W = self.grid_shape
        a = u1.float().view(-1, 1, H, W)
        b = u2.float().view(-1, 1, H, W)
        return torch.cat([a, b], dim=1)

    def _standardize(self, data):
        mean, std = self.stats
        return (data - mean.to(data.device)) / std.to(data.device)

    def fit_transform_dicts(self, dict_train, input_keys=INPUT_KEYS, dict_val=None):
        u1, u2 = dict_train["u1_in"], dict_train["u2_in"]
        data = self._stack(u1, u2)

        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
        self.stats = (mean, std)
        data_scaled = self._standardize(data)

        val_loader = None
        if dict_val is not None and "u1_in" in dict_val and "u2_in" in dict_val:
            data_val = self._standardize(self._stack(dict_val["u1_in"], dict_val["u2_in"]))
            val_loader = DataLoader(TensorDataset(data_val), batch_size=self.batch_size, shuffle=False)

        self.model = JointConvAutoencoder(latent_dim=self.latent_dim, base=self.base).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(data_scaled), batch_size=self.batch_size, shuffle=True)

        print(f"\n[joint AE] Training (latent d={self.latent_dim}, 2 channels) on {self.device} ...")
        best_val, best_state = float("inf"), None
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                rec, _ = self.model(batch)
                loss = F.mse_loss(rec, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_mse = epoch_loss / len(loader)

            val_mse = None
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    v = sum(F.mse_loss(self.model(vb.to(self.device))[0], vb.to(self.device)).item()
                            for (vb,) in val_loader)
                val_mse = v / len(val_loader)
                if val_mse < best_val:
                    best_val = val_mse
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  [joint AE] Epoch {epoch+1:3d}/{self.epochs} | Train MSE: {train_mse:.6f}"
                if val_mse is not None:
                    msg += f" | Val MSE: {val_mse:.6f} | Best Val: {best_val:.6f}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [joint AE] Best model reloaded (Val MSE = {best_val:.6f})")
        self.model.eval()
        self.fitted = True

        return self.transform_dicts(dict_train, input_keys)

    def transform_dicts(self, dict_in, input_keys=INPUT_KEYS):
        """Return the joint latent z under BOTH u1_in and u2_in (other keys untouched)."""
        if not self.fitted:
            raise ValueError("JointAutoencoderEncoder must be fitted (train_encoder) first.")
        z = self._encode(dict_in["u1_in"], dict_in["u2_in"])
        out = dict_in.copy()
        out["u1_in"] = z
        out["u2_in"] = z
        return out

    def _encode(self, u1, u2):
        """(N, N_cells) x2 -> z (N, latent_dim), in chunks."""
        self.model.eval()
        data = self._standardize(self._stack(u1, u2))
        outs = []
        with torch.no_grad():
            for i in range(0, data.shape[0], self.encode_batch):
                chunk = data[i:i + self.encode_batch].to(self.device)
                outs.append(self.model.encode(chunk).cpu())
        return torch.cat(outs, dim=0).float()

    def reconstruct_fields(self, u1, u2):
        """Rebuild BOTH fields (T_hat, kappa_hat) at physical scale, (N, N_cells) each."""
        self.model.eval()
        data = self._standardize(self._stack(u1, u2))
        outs = []
        with torch.no_grad():
            for i in range(0, data.shape[0], self.encode_batch):
                chunk = data[i:i + self.encode_batch].to(self.device)
                outs.append(self.model(chunk)[0].cpu())
        rec = torch.cat(outs, dim=0)
        mean, std = self.stats
        rec = rec * std.to(rec.device) + mean.to(rec.device)
        n = rec.shape[0]
        return rec[:, 0].reshape(n, -1).float(), rec[:, 1].reshape(n, -1).float()

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mean, std = self.stats
        state = {
            "kind": self.kind, "latent_dim": self.latent_dim, "base": self.base,
            "grid_shape": self.grid_shape, "epochs": self.epochs,
            "batch_size": self.batch_size, "lr": self.lr,
            "latent_dims": self.latent_dims,
            "model": self.model.state_dict(),
            "stats": (mean.cpu(), std.cpu()),
        }
        torch.save(state, path)
        print(f"Joint encoder saved to: {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device)
        enc = cls(latent_dim=state["latent_dim"], grid_shape=state["grid_shape"],
                  base=state.get("base", 32), epochs=state.get("epochs", 600),
                  batch_size=state.get("batch_size", 64), lr=state.get("lr", 1e-3),
                  device=device)
        enc.latent_dims = state["latent_dims"]
        enc.model = JointConvAutoencoder(latent_dim=state["latent_dim"],
                                         base=state.get("base", 32)).to(device)
        enc.model.load_state_dict(state["model"])
        enc.model.eval()
        mean, std = state["stats"]
        enc.stats = (mean.to(device), std.to(device))
        enc.fitted = True
        return enc


def joint_encoder_path(name, tag="ae", save_dir=CHECKPOINTS_DIR):
    """Path of the joint encoder: checkpoints/encoder_<tag>_<name>.pt for the
    Transformer-Based DeepONet, checkpoints/encoder_joint_ae_<name>.pt for the
    convolutional one.

    As for the fixed-rank encoders, the head suffix is stripped: `TBD-N1-D128-affine`
    reuses the encoder of `TBD-N1-D128` instead of refitting one (cf.
    `encoders.encoder_run`)."""
    from encoders import encoder_run
    stem = f"encoder_{tag}" if tag == "tbd" else f"encoder_joint_{tag}"
    return os.path.join(save_dir, f"{stem}_{encoder_run(name)}.pt")


def _joint_encoder_tag(args):
    return "tbd" if getattr(args, "joint_encoder", "conv") == "tbd" else "ae"


def operator_path(name, save_dir=CHECKPOINTS_DIR):
    """Path of the (T,kappa)->I operator with Transformer-Based DeepONet decoder:
    checkpoints/operator_tbd_<name>.pt."""
    return os.path.join(save_dir, f"operator_tbd_{name}.pt")


class SingleBranchDeepONet(nn.Module):
    """Single-branch 2D RTE DeepONet: I = softplus(branch(z).trunk(y) + b)*I_scale.

    branch_inputs is a list so the signature stays compatible with Trainer/Evaluator
    (which pass [u1, u2]); only branch_inputs[0] = the joint latent z is used.
    """

    def __init__(self, branch_input_dim, trunk_input_dim, p_basis,
                 branch_hidden=[512, 512, 512, 512], trunk_hidden=[512, 512, 512, 512],
                 I_scale_val=1.0):
        super().__init__()
        self.p_basis = p_basis
        self.branch = MLP(branch_input_dim, p_basis, branch_hidden)
        self.trunk = MLP(trunk_input_dim, p_basis, trunk_hidden)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("I_scale", torch.tensor(I_scale_val, dtype=torch.float32))

    def forward(self, branch_inputs, trunk_input):
        c = self.branch(branch_inputs[0])
        trunk_output = self.trunk(trunk_input)

        if trunk_input.dim() == 2:
            raw_out = torch.matmul(c, trunk_output.T) + self.bias
        elif trunk_input.dim() == 3:
            raw_out = torch.sum(c.unsqueeze(1) * trunk_output, dim=-1) + self.bias
        else:
            raise ValueError(f"Unsupported trunk_input rank: {trunk_input.dim()}")
        return F.softplus(raw_out) * self.I_scale

    def pde_residual(self, branch_inputs, y, u1_eval, u2_eval):
        """2D RTE residual -- same structure as the other models (c does not depend on y)."""
        c = self.branch(branch_inputs[0])

        t = self.trunk(y)
        raw = torch.matmul(c, t.T) + self.bias
        I = F.softplus(raw) * self.I_scale

        trunk_params = {**dict(self.trunk.named_parameters()),
                        **dict(self.trunk.named_buffers())}

        def _trunk_single(params, y_single):
            return functional_call(self.trunk, params, (y_single.unsqueeze(0),)).squeeze(0)

        J = vmap(jacfwd(_trunk_single, argnums=1), in_dims=(None, 0))(trunk_params, y)

        sig = torch.sigmoid(raw)
        dI_dx = self.I_scale * sig * torch.matmul(c, J[:, :, 0].T)
        dI_dy = self.I_scale * sig * torch.matmul(c, J[:, :, 1].T)

        sin_phi, cos_phi = transport_coeffs(y[:, 2])
        transport = sin_phi * dI_dx + cos_phi * dI_dy
        residual = transport - (u1_eval + u2_eval * I)
        return residual, I


def infer_architecture_single(state_dict):
    """Infer (branch_input_dim, trunk_input_dim, p_basis, hidden) from the .pth."""
    b_idx = sorted({int(k.split(".")[2]) for k in state_dict
                    if k.startswith("branch.network.") and k.endswith(".weight")})
    b_w = [state_dict[f"branch.network.{i}.weight"] for i in b_idx]
    branch_input_dim = b_w[0].shape[1]
    p_basis = b_w[-1].shape[0]
    branch_hidden = [w.shape[0] for w in b_w[:-1]]

    t_idx = sorted({int(k.split(".")[2]) for k in state_dict
                    if k.startswith("trunk.network.") and k.endswith(".weight")})
    t_w = [state_dict[f"trunk.network.{i}.weight"] for i in t_idx]
    trunk_input_dim = t_w[0].shape[1]
    trunk_hidden = [w.shape[0] for w in t_w[:-1]]
    return dict(branch_input_dim=branch_input_dim, trunk_input_dim=trunk_input_dim,
                p_basis=p_basis, branch_hidden=branch_hidden, trunk_hidden=trunk_hidden)


def load_model_single(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    arch = infer_architecture_single(state_dict)
    print(f"Single-branch architecture inferred from {model_path}:")
    print(f"  branch_input_dim={arch['branch_input_dim']} | p_basis={arch['p_basis']} "
          f"| branch_hidden={arch['branch_hidden']} | trunk_hidden={arch['trunk_hidden']}")
    model = SingleBranchDeepONet(**arch).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  I_scale = {model.I_scale.item():.4e}")
    return model


def _build_joint_encoder(args):
    """Instantiate the joint 2-channel encoder: conv (JointConvAE) or tbd (FunDiff)."""
    if getattr(args, "joint_encoder", "conv") == "tbd":
        from tbdeeponet import TBDEncoder, parse_subsample_arg
        return TBDEncoder(
            epochs=args.ae_epochs, batch_size=args.ae_batch_size, lr=args.ae_lr,
            patch=args.tbd_patch, dim=args.tbd_dim, n_latent=args.tbd_n_latent,
            depth=args.tbd_depth, dec_depth=args.tbd_dec_depth, heads=args.tbd_heads,
            n_fourier=args.tbd_n_fourier, fourier_sigma=args.tbd_sigma, n_query=args.tbd_n_query,
            subsample_factors=parse_subsample_arg(getattr(args, "tbd_subsample", "auto")))
    return JointAutoencoderEncoder(
        latent_dim=args.ae_latent_dim, epochs=args.ae_epochs,
        batch_size=args.ae_batch_size, lr=args.ae_lr)


def run_train_encoder(args):
    print(f"\n--- MODE TRAIN_ENCODER (joint {getattr(args, 'joint_encoder', 'conv')}) ---")
    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    data_train, data_val = splits["data"]

    encoder = _build_joint_encoder(args)
    encoder.fit_transform_dicts(
        {"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]}, INPUT_KEYS,
        dict_val={"u1_in": data_val["u1_in"], "u2_in": data_val["u2_in"]},
        resume_name=args.name if args.resume else None)

    path = joint_encoder_path(args.name, _joint_encoder_tag(args))
    encoder.save(path)
    print(f"Joint latent size: {encoder.latent_dim}")
    print(f"[OK] Encoder ready. Run: --mode train_deeponet --name {args.name}"
          + (f" --joint_encoder {args.joint_encoder}" if getattr(args, 'joint_encoder', 'conv') != 'conv' else ""))


def _load_joint_encoder(args, device):
    path = joint_encoder_path(args.name, _joint_encoder_tag(args))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Joint encoder not found: {path}\n"
            f"Run first: python {os.path.basename(__file__)} --mode train_encoder "
            f"--name {args.name}"
            + (f" --joint_encoder {args.joint_encoder}" if getattr(args, 'joint_encoder', 'conv') != 'conv' else ""))
    print(f"Loading joint encoder: {path}")
    if getattr(args, "joint_encoder", "conv") == "tbd":
        from tbdeeponet import TBDEncoder
        return TBDEncoder.load(path, device=device)
    return JointAutoencoderEncoder.load(path, device=device)


def run_train_deeponet(args, device):
    print(f"\n--- MODE TRAIN_DEEPONET (mono-branche, encodeur joint) ---")
    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    dict_pde_train, dict_pde_val = splits["pde"]
    dict_bc_train, dict_bc_val = splits["bc"]
    data_train, data_val = splits["data"]
    y_full = splits["y_full"]

    encoder = _load_joint_encoder(args, device)

    dict_pde_train = encoder.transform_dicts(dict_pde_train, INPUT_KEYS)
    dict_pde_val = encoder.transform_dicts(dict_pde_val, INPUT_KEYS)
    dict_bc_train = encoder.transform_dicts(dict_bc_train, INPUT_KEYS)
    dict_bc_val = encoder.transform_dicts(dict_bc_val, INPUT_KEYS)
    enc_data_train = encoder.transform_dicts(
        {"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]}, INPUT_KEYS)
    enc_data_val = encoder.transform_dicts(
        {"u1_in": data_val["u1_in"], "u2_in": data_val["u2_in"]}, INPUT_KEYS)

    colloc_sampler = None
    if args.n_colloc > 0:
        n_cells = data_train["u1_in"].shape[1]
        n_dirs = y_full.shape[0] // n_cells
        cell_xy = y_full[::n_dirs, :2]
        colloc_sampler = CollocationSampler(cell_xy)
        dict_pde_train["T_raw"], dict_pde_train["kappa_raw"] = data_train["u1_in"], data_train["u2_in"]
        dict_pde_val["T_raw"], dict_pde_val["kappa_raw"] = data_val["u1_in"], data_val["u2_in"]
        print(f"{args.n_colloc} collocation points resampled per epoch "
              f"({n_cells} cells, {n_dirs} directions).")

    train_dataset = FullGridDataset(enc_data_train["u1_in"], enc_data_train["u2_in"], data_train["I_target"], y_full)
    val_dataset = FullGridDataset(enc_data_val["u1_in"], enc_data_val["u2_in"], data_val["I_target"], y_full)
    print(f"FullGridDataset built: {len(train_dataset)} train snapshots, {len(val_dataset)} val.")

    print(f"Joint latent size (single branch net): {encoder.latent_dim}")

    I_target_train = data_train["I_target"]
    I_non_zero = I_target_train[I_target_train > 1e-3]
    median_I = torch.median(I_non_zero).item() if len(I_non_zero) > 0 else 1.0
    print(f"Computed I_scale (median): {median_I:.4e}")

    model = SingleBranchDeepONet(
        branch_input_dim=encoder.latent_dim,
        trunk_input_dim=4,
        p_basis=args.p_basis,
        trunk_hidden=[512, 512, 512, 512],
        I_scale_val=median_I,
    ).to(device)

    trainer = Trainer(model, device, save_dir=CHECKPOINTS_DIR)
    trainer.train(
        dict_pde_train, dict_bc_train, train_dataset,
        dict_pde_val, dict_bc_val, val_dataset,
        epochs=args.epochs, batch_size_pde=args.batch_size_pde,
        batch_size_data=args.batch_size_data,
        w_phys=args.w_phys, w_bc=args.w_bc, w_data=args.w_data,
        save_name=args.name, n_colloc=args.n_colloc, colloc_sampler=colloc_sampler,
        loss_balancing=args.loss_balancing, relobralo_alpha=args.relobralo_alpha,
        relobralo_temp=args.relobralo_temp, relobralo_rho=args.relobralo_rho,
    )


def _resolve_snapshot_indices(spec, data_val):
    """Parse `--eval_snapshots`: non-negative integers = direct index into data_val;
    -1 = the MOST INTENSE snapshot; "chaotic[:k]" = the k snapshots with the largest
    spatial variance of G. Delegated to `utils.resolve_snapshots` so the CLI and the
    plotting helpers share EXACTLY the same convention."""
    from utils import resolve_snapshots
    return resolve_snapshots(spec, data_val)


def run_eval_2d(args, device):
    print(f"\n--- MODE EVAL_2D (mono-branche, encodeur joint) ---")
    encoder = _load_joint_encoder(args, device)
    print(f"Encoder loaded. Joint latent size: {encoder.latent_dim}")

    model_path = args.model_path or os.path.join(CHECKPOINTS_DIR, f"{args.name}.pth")
    model = load_model_single(model_path, device)

    evaluator = Evaluator(model, device, h5_path=solver_h5())
    evaluator.encoder = encoder

    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    data_eval = split_data(splits, args.eval_split)
    y_full = splits["y_full"]

    print(f"\n--- Comparison maps (from the .pt; raw solver H5 unavailable) "
          f"-- {args.eval_split.upper()} split ---")
    plot_eval_snapshots(evaluator, data_eval, y_full, spec=args.eval_snapshots,
                        output_dir=args.output_dir, include_unseen_angles=False)

    print(f"\n--- Errors on the intensity I ({args.eval_split.upper()} split) ---")
    enc_eval = encoder.transform_dicts(
        {"u1_in": data_eval["u1_in"], "u2_in": data_eval["u2_in"]}, INPUT_KEYS)
    eval_loader = DataLoader(
        TensorDataset(enc_eval["u1_in"], enc_eval["u2_in"], data_eval["I_target"]),
        batch_size=args.batch_size_data, shuffle=False)
    g_ref = None
    if getattr(args, "g_ref", "integrated") == "solver":
        from utils import solver_G
        g_ref = solver_G(splits[f"{args.eval_split}_idx"])
        print(f"   G reference: the solver's own G field ({tuple(g_ref.shape)})")
    return evaluator.evaluate_split_errors(eval_loader, y_coords=y_full,
                                           label=args.eval_split.upper(), g_ref=g_ref)


def _fresh_encoder_for_operator(args, data_train, device):
    """RANDOM (from-scratch) Transformer-Based DeepONet encoder: input stats computed on
    the train split, the AE itself untrained (it will be trained jointly with dec_I)."""
    from tbdeeponet import TBDEncoder
    enc = TBDEncoder(
        patch=args.tbd_patch, dim=args.tbd_dim, n_latent=args.tbd_n_latent,
        depth=args.tbd_depth, dec_depth=args.tbd_dec_depth, heads=args.tbd_heads,
        n_fourier=args.tbd_n_fourier, fourier_sigma=args.tbd_sigma, n_query=args.tbd_n_query,
        device=device)
    data = enc._stack(data_train["u1_in"], data_train["u2_in"])
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std = data.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    enc.stats = (mean.to(device), std.to(device))
    enc.model = enc._make_model().to(device)
    enc.fitted = True
    return enc


def _resolve_operator_encoder(args, data_train, device):
    """Return (encoder, is_pretrained). `--pretrained_encoder` accepts: a .pt PATH, an
    encoder NAME (checkpoints/encoder_tbd_<name>.pt), 'auto' (= the operator's
    name), or 'scratch'/'none' (random from-scratch encoder)."""
    from tbdeeponet import TBDEncoder
    src = args.pretrained_encoder
    if src in ("scratch", "none", "off"):
        print("[operator] FROM-SCRATCH encoder (random, trained together with dec_I).")
        return _fresh_encoder_for_operator(args, data_train, device), False
    if src in (None, "auto"):
        path = joint_encoder_path(args.name, "tbd")
    elif os.path.exists(src):
        path = src
    else:
        path = joint_encoder_path(src, "tbd")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained encoder not found: {path}\n"
            f"Train it first (--mode train_encoder --joint_encoder tbd), or pass "
            f"--pretrained_encoder scratch to train from scratch.")
    print(f"[operator] Pretrained Transformer-Based DeepONet encoder: {path}")
    return TBDEncoder.load(path, device=device), True


def run_train_operator(args, device):
    from tbdeeponet import TBDOperator, fit_operator, save_operator
    print("\n--- MODE TRAIN_OPERATOR (Transformer-Based DeepONet coordinate decoder -> I) ---")
    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    data_train, data_val = splits["data"]
    y_full = splits["y_full"]

    if args.op_data_frac < 1.0:
        n_total = data_train["u1_in"].shape[0]
        n_keep = max(1, int(round(args.op_data_frac * n_total)))
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(n_total, generator=gen)[:n_keep]
        data_train = {k: v[perm] for k, v in data_train.items()}
        print(f"[operator] Train subsampling: {args.op_data_frac:.0%} "
              f"-> {n_keep}/{n_total} snapshots (val UNCHANGED, {data_val['u1_in'].shape[0]} snapshots)")

    encoder, is_pretrained = _resolve_operator_encoder(args, data_train, device)
    freeze = args.freeze_encoder and is_pretrained
    if args.freeze_encoder and not is_pretrained:
        print("[operator] From-scratch: a random encoder cannot be frozen -> unfrozen.")

    I_train = data_train["I_target"]
    I_nz = I_train[I_train > 1e-3]
    median_I = torch.median(I_nz).item() if len(I_nz) > 0 else 1.0
    print(f"I_scale (median of I>1e-3) = {median_I:.4e}")

    dec_depth = args.op_dec_depth if args.op_dec_depth > 0 else encoder.dec_depth
    operator = TBDOperator(
        encoder.model, I_scale=median_I, dec_depth=dec_depth, heads=encoder.heads,
        mlp_ratio=encoder.mlp_ratio, n_fourier=encoder.n_fourier,
        fourier_sigma=encoder.fourier_sigma, affine=args.affine).to(device)
    operator.set_coord_norm(y_full)
    if args.init_decoder_from_recon and is_pretrained:
        operator.init_dec_I_from_recon()

    print(f"Operator: encoder {'FROZEN' if freeze else 'FINE-TUNED'}, latent {encoder.latent_dim} "
          f"(n_latent={encoder.n_latent} x dim={encoder.dim}), dec_I depth={dec_depth}, "
          f"w_recon={args.w_recon}, w_phys={args.w_phys}, w_bc={args.w_bc}, "
          f"loss_balancing={args.loss_balancing}, data_loss={args.data_loss}, "
          f"query=(x, y, phi), "
          f"head={'AFFINE (no softplus)' if args.affine else 'softplus'}")
    path = operator_path(args.name)
    fit_operator(
        operator, encoder,
        {"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]},
        {"u1_in": data_val["u1_in"], "u2_in": data_val["u2_in"]},
        y_full, I_train, data_val["I_target"],
        epochs=args.op_epochs, batch_size=args.op_batch, lr=args.op_lr,
        n_query=args.op_n_query, w_recon=args.w_recon, freeze_encoder=freeze, device=device,
        val_every=args.op_val_every, encoder_lr=args.op_encoder_lr, save_path=path,
        w_phys=args.w_phys, w_bc=args.w_bc, w_data=args.w_data, n_query_phys=args.op_n_query_phys,
        loss_balancing=args.loss_balancing, relobralo_alpha=args.relobralo_alpha,
        relobralo_temp=args.relobralo_temp, relobralo_rho=args.relobralo_rho,
        monitor_phys=args.monitor_phys, history_name=args.name, resume=args.resume,
        data_loss=args.data_loss)

    save_operator(operator, encoder, path,
                  meta={"pretrained": str(args.pretrained_encoder), "freeze": freeze,
                        "w_recon": args.w_recon, "from_scratch": not is_pretrained,
                        "w_phys": args.w_phys, "w_bc": args.w_bc,
                        "loss_balancing": args.loss_balancing,
                        "data_loss": args.data_loss, "query": "xy_phi"})
    print(f"[OK] Operator ready. Evaluate with: python {os.path.basename(__file__)} "
          f"--mode eval_operator --name {args.name}")


def run_eval_operator(args, device):
    from tbdeeponet import load_operator
    print("\n--- MODE EVAL_OPERATOR (Transformer-Based DeepONet decoder -> I) ---")
    path = args.model_path or operator_path(args.name)
    operator, encoder = load_operator(path, device=device)
    print(f"Operator loaded: {path} | latent {encoder.latent_dim} | I_scale={float(operator.I_scale):.4e}")

    evaluator = Evaluator(operator, device, h5_path=solver_h5())
    evaluator.encoder = encoder

    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    data_eval = split_data(splits, args.eval_split)
    y_full = splits["y_full"]

    print(f"\n--- Comparison maps (from the .pt; raw solver H5 unavailable) "
          f"-- {args.eval_split.upper()} split ---")
    plot_eval_snapshots(evaluator, data_eval, y_full, spec=args.eval_snapshots,
                        output_dir=args.output_dir, include_unseen_angles=False)

    print(f"\n--- Errors on I ({args.eval_split.upper()} split) ---")
    enc_eval = encoder.transform_dicts(
        {"u1_in": data_eval["u1_in"], "u2_in": data_eval["u2_in"]}, INPUT_KEYS)
    eval_loader = DataLoader(
        TensorDataset(enc_eval["u1_in"], enc_eval["u2_in"], data_eval["I_target"]),
        batch_size=args.batch_size_data, shuffle=False)
    g_ref = None
    if getattr(args, "g_ref", "integrated") == "solver":
        from utils import solver_G
        g_ref = solver_G(splits[f"{args.eval_split}_idx"])
        print(f"   G reference: the solver's own G field ({tuple(g_ref.shape)})")
    return evaluator.evaluate_split_errors(eval_loader, y_coords=y_full,
                                           label=args.eval_split.upper(), g_ref=g_ref)


def build_parser():
    p = argparse.ArgumentParser(description="2D RTE DeepONet -- single branch + joint autoencoder")
    p.add_argument("--mode", type=str, required=True,
                   choices=["train_encoder", "train_deeponet", "eval_2d",
                            "train_operator", "eval_operator"])
    p.add_argument("--name", type=str, default="deeponet_joint")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size_pde", type=int, default=512)
    p.add_argument("--batch_size_data", type=int, default=256)
    p.add_argument("--w_phys", type=float, default=0.0)
    p.add_argument("--w_bc", type=float, default=0.0)
    p.add_argument("--w_data", type=float, default=1.0)
    p.add_argument("--n_colloc", type=int, default=50000)
    p.add_argument("--p_basis", type=int, default=128)
    p.add_argument("--loss_balancing", type=str, default="static", choices=["static", "relobralo"])
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Resume an interrupted run from "
                        "checkpoints/<name>_{enc,op}_resume.pt (default). --no-resume to "
                        "start over, ignoring the saved state.")
    p.add_argument("--monitor_phys", action=argparse.BooleanOptionalAction, default=True,
                   help="Measure the physics loss (train+val) at the validation cadence "
                        "EVEN in data-only mode (w_phys=0), so it can be plotted: it costs "
                        "about 0.01 %% of the epoch time. --no-monitor_phys to switch off.")
    p.add_argument("--relobralo_alpha", type=float, default=0.99)
    p.add_argument("--relobralo_temp", type=float, default=0.1)
    p.add_argument("--relobralo_rho", type=float, default=0.999)

    p.add_argument("--joint_encoder", type=str, default="conv", choices=["conv", "tbd"],
                   help="Joint 2-channel encoder: 'conv' (JointConvAE, default) or "
                        "'tbd' (FunDiff ViT+Perceiver, coordinate decoder)")
    p.add_argument("--ae_latent_dim", type=int, default=512, help="Joint latent size (conv)")
    p.add_argument("--ae_epochs", type=int, default=600)
    p.add_argument("--ae_batch_size", type=int, default=64)
    p.add_argument("--ae_lr", type=float, default=1e-3)

    p.add_argument("--tbd_patch", type=int, default=10)
    p.add_argument("--tbd_dim", type=int, default=128)
    p.add_argument("--tbd_n_latent", type=int, default=4, help="joint latent = n_latent*dim (default 4*128=512)")
    p.add_argument("--tbd_depth", type=int, default=4)
    p.add_argument("--tbd_dec_depth", type=int, default=4)
    p.add_argument("--tbd_heads", type=int, default=8)
    p.add_argument("--tbd_n_fourier", type=int, default=64)
    p.add_argument("--tbd_sigma", type=float, default=3.0)
    p.add_argument("--tbd_n_query", type=int, default=4096)
    p.add_argument("--tbd_subsample", type=str, default="auto",
                   help="Random input subsampling during training (resolution invariance, "
                        "cf. FunDiff): 'auto', 'none', or a list such as '1,3,5'")

    p.add_argument("--pretrained_encoder", type=str, default="auto",
                   help="Starting Transformer-Based DeepONet encoder: a .pt PATH, an "
                        "encoder NAME, 'auto' "
                        "(= --name), or 'scratch' (random, from-scratch encoder).")
    p.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze the encoder while dec_I trains (default). "
                        "--no-freeze_encoder to fine-tune it.")
    p.add_argument("--w_recon", type=float, default=0.0,
                   help="Weight of the auxiliary (T,kappa) reconstruction loss -- only "
                        "when the encoder is unfrozen (0 = loss on I alone).")
    p.add_argument("--op_epochs", type=int, default=1000)
    p.add_argument("--op_lr", type=float, default=1e-3)
    p.add_argument("--op_batch", type=int, default=64, help="snapshots per step")
    p.add_argument("--op_n_query", type=int, default=8192,
                   help="RTE points (cell x direction) subsampled per step")
    p.add_argument("--op_n_query_phys", type=int, default=None,
                   help="physics collocation points per step (w_phys>0; default = op_n_query)")
    p.add_argument("--op_dec_depth", type=int, default=0,
                   help="depth of dec_I (0 = same as the encoder's decoder)")
    p.add_argument("--affine", action="store_true",
                   help="Drop the output softplus: I = D_I(z, y) * I_scale. The decoder "
                        "stays a cross-attention, so the reconstructor is NOT the R of "
                        "Thm 3.4 -- this only isolates what the softplus itself buys.")
    p.add_argument("--op_val_every", type=int, default=25,
                   help="validate every N epochs (full-angle validation is expensive)")
    p.add_argument("--data_loss", type=str, default="rel_l2", choices=["rel_l2", "mse"],
                   help="Operator data loss: rel_l2 = per-snapshot relative L2 "
                        "(Lu et al.; the default here); mse = pooled MSE / I_scale^2")
    p.add_argument("--op_encoder_lr", type=float, default=None,
                   help="Encoder LR when unfrozen (None = same as dec_I; use something "
                        "smaller to gently fine-tune a pretrained encoder, e.g. 1e-4)")
    p.add_argument("--init_decoder_from_recon", action="store_true",
                   help="Warm-start the cross-attention blocks of dec_I from the recon decoder.")
    p.add_argument("--op_data_frac", type=float, default=1.0,
                   help="Fraction of the TRAIN split to use (data-efficiency study): "
                        "1.0 = all (default), 0.1/0.3 = 10%%/30%%. The val split stays "
                        "UNCHANGED. Nested subsets (fixed seed): 10%% in 30%% in 100%%.")

    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Evaluation output directory (default: results/<name>/)")
    p.add_argument("--g_ref", type=str, default="integrated",
        choices=("integrated", "solver"),
        help="Reference for G: the integrated sum_m I_target (default), or the solver's own G field read from RTE2D_SOLVER_H5.")
    p.add_argument("--eval_split", type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Split eval_2d / eval_operator measure and plot on. TEST by "
                        "default: the val split was used to pick the checkpoint.")
    p.add_argument("--eval_snapshots", type=str, default=SNAPSHOT_SPEC_DEFAULT,
                   help="Snapshots for the comparison maps, comma separated: direct index "
                        "('0,20,100'), -1 = the most intense one, 'structure[:k]' = the k most "
                        "structured, 'chaotic[:k]' = the k of largest spatial variance of G, "
                        "'random[:k]' = k drawn at random (fixed seed, hence the same "
                        "snapshots for every model). "
                        f"Default: '{SNAPSHOT_SPEC_DEFAULT}' = 8 snapshots.")
    return p


def parse_arguments(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments()
    if args.output_dir is None:
        args.output_dir = os.path.join(RESULTS_DIR, args.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== RUNNING ON: {device.type.upper()} ===")

    if args.mode == "train_encoder":
        run_train_encoder(args)
    elif args.mode == "train_deeponet":
        run_train_deeponet(args, device)
    elif args.mode == "eval_2d":
        run_eval_2d(args, device)
    elif args.mode == "train_operator":
        run_train_operator(args, device)
    elif args.mode == "eval_operator":
        run_eval_operator(args, device)
