import argparse
import os
import pickle
from itertools import cycle

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call, jacfwd, vmap
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial import cKDTree

from encoders import make_encoder, load_encoder, encoder_path, encoder_run
from utils import (Evaluator, load_and_split_dataset, resolve_snapshots, split_data, solver_h5,
                   plot_eval_snapshots, SNAPSHOT_SPEC_DEFAULT, transport_coeffs,
                   QUERY_DIM, DEFAULT_DATASET, MESH_STUB, CHECKPOINTS_DIR, RESULTS_DIR,
                   save_figure)

INPUT_KEYS = ["u1_in", "u2_in"]

SIGMA_SB = 5.670374419e-8
IB_COEF = SIGMA_SB / np.pi


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation=nn.GELU(),
                 activate_last=False):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or activate_last:
                layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiBranchDeepONet(nn.Module):
    """Multi-input DeepONet: I = act(<beta, gamma(y)> + b) * I_scale.

    act is softplus, or the identity with `affine=True` (the affine class of Thm 3.4)."""

    def __init__(self, branch_input_dims, trunk_input_dim, p_basis,
                 branch_hidden=[512, 512, 512, 512], trunk_hidden=[512, 512, 512, 512],
                 I_scale_val=1.0, trunk_activate_last=False, affine=False):
        super().__init__()
        self.branches = nn.ModuleList([
            MLP(dim, p_basis, branch_hidden) for dim in branch_input_dims
        ])
        self.trunk = MLP(trunk_input_dim, p_basis, trunk_hidden,
                         activate_last=trunk_activate_last)
        self.trunk_input_dim = trunk_input_dim
        self.bias = nn.Parameter(torch.zeros(1))

        self.register_buffer("I_scale", torch.tensor(I_scale_val, dtype=torch.float32))
        if trunk_activate_last:
            self.register_buffer("trunk_act_last", torch.tensor(1.0))
        self.affine = affine
        if affine:
            self.register_buffer("affine_head", torch.tensor(1.0))

    def forward(self, branch_inputs, trunk_input):
        branch_outputs = [branch(b_in) for branch, b_in in zip(self.branches, branch_inputs)]
        combined_branches = branch_outputs[0]
        for b_out in branch_outputs[1:]:
            combined_branches = combined_branches * b_out

        trunk_output = self.trunk(trunk_input)

        if trunk_input.dim() == 2:
            raw_out = (torch.matmul(combined_branches, trunk_output.T)) + self.bias
        elif trunk_input.dim() == 3:
            raw_out = (torch.sum(combined_branches.unsqueeze(1) * trunk_output, dim=-1)) + self.bias
        else:
            raise ValueError(f"Unsupported trunk_input rank: {trunk_input.dim()}")

        out = raw_out if self.affine else torch.nn.functional.softplus(raw_out)
        return out * self.I_scale

    def pde_residual(self, branch_inputs, y, u1_eval, u2_eval):
        """RTE residual per snapshot, the trunk Jacobian computed once and shared.

        branch_inputs: latents [(B, d1), (B, d2)]; y: (P, 3) = (x, y, phi);
        u1_eval = kappa*Ib, u2_eval = -kappa, both (B, P).
        Returns (residual, I), both (B, P)."""
        c = self.branches[0](branch_inputs[0])
        for branch, b_in in zip(self.branches[1:], branch_inputs[1:]):
            c = c * branch(b_in)

        t = self.trunk(y)
        raw = torch.matmul(c, t.T) + self.bias
        I = (raw if self.affine else F.softplus(raw)) * self.I_scale

        trunk_params = {**dict(self.trunk.named_parameters()),
                        **dict(self.trunk.named_buffers())}

        def _trunk_single(params, y_single):
            return functional_call(self.trunk, params, (y_single.unsqueeze(0),)).squeeze(0)

        J = vmap(jacfwd(_trunk_single, argnums=1), in_dims=(None, 0))(trunk_params, y)

        sig = 1.0 if self.affine else torch.sigmoid(raw)
        dI_dx = self.I_scale * sig * torch.matmul(c, J[:, :, 0].T)
        dI_dy = self.I_scale * sig * torch.matmul(c, J[:, :, 1].T)

        sin_phi, cos_phi = transport_coeffs(y[:, 2])
        transport = sin_phi * dI_dx + cos_phi * dI_dy
        residual = transport - (u1_eval + u2_eval * I)
        return residual, I


def infer_architecture(state_dict):
    """Infer (branch_input_dims, trunk_input_dim, p_basis, branch_hidden, trunk_hidden)."""
    branch_ids = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("branches.")})
    branch_input_dims, branch_hidden, p_basis = [], None, None
    for b in branch_ids:
        lin_idx = sorted({int(k.split(".")[3]) for k in state_dict
                          if k.startswith(f"branches.{b}.network.") and k.endswith(".weight")})
        weights = [state_dict[f"branches.{b}.network.{i}.weight"] for i in lin_idx]
        branch_input_dims.append(weights[0].shape[1])
        hidden = [w.shape[0] for w in weights[:-1]]
        p_basis = weights[-1].shape[0]
        if branch_hidden is None:
            branch_hidden = hidden

    t_idx = sorted({int(k.split(".")[2]) for k in state_dict
                    if k.startswith("trunk.network.") and k.endswith(".weight")})
    t_w = [state_dict[f"trunk.network.{i}.weight"] for i in t_idx]
    trunk_input_dim = t_w[0].shape[1]
    trunk_hidden = [w.shape[0] for w in t_w[:-1]]
    return branch_input_dims, trunk_input_dim, p_basis, branch_hidden, trunk_hidden


def load_model(model_path, device):
    """Rebuild the MultiBranchDeepONet from the .pth alone and load the weights."""
    state_dict = torch.load(model_path, map_location=device)
    branch_dims, trunk_in, p_basis, branch_hidden, trunk_hidden = infer_architecture(state_dict)
    trunk_act = "trunk_act_last" in state_dict
    affine = "affine_head" in state_dict
    print(f"Architecture inferred from {model_path}:")
    print(f"  branch_input_dims = {branch_dims} | trunk_input_dim = {trunk_in} | p_basis = {p_basis}")
    print(f"  branch_hidden = {branch_hidden} | trunk_hidden = {trunk_hidden} | trunk_activate_last = {trunk_act}")
    print(f"  head = {'AFFINE (no softplus, Thm 3.4)' if affine else 'softplus'}")
    model = MultiBranchDeepONet(
        branch_input_dims=branch_dims, trunk_input_dim=trunk_in, p_basis=p_basis,
        branch_hidden=branch_hidden, trunk_hidden=trunk_hidden,
        trunk_activate_last=trunk_act, affine=affine,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  I_scale = {model.I_scale.item():.4e}")
    return model


class FullGridDataset(torch.utils.data.Dataset):
    def __init__(self, u1_in, u2_in, I_target_full, y_full):
        self.u1_in = u1_in
        self.u2_in = u2_in
        self.I_target_full = I_target_full
        self.y_full = y_full
        self.num_snapshots = u1_in.shape[0]

    def __len__(self):
        return self.num_snapshots

    def __getitem__(self, snapshot_idx):
        u1 = self.u1_in[snapshot_idx]
        u2 = self.u2_in[snapshot_idx]
        I_target = self.I_target_full[snapshot_idx]
        return u1, u2, I_target


class DeviceTensorLoader:
    """Mini-loader over tensors moved to the device once and sliced by index.

    The whole split must fit in GPU memory."""

    def __init__(self, tensors, batch_size, shuffle, device):
        self.tensors = [t.to(device) for t in tensors]
        self.n = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = (torch.randperm(self.n, device=self.device) if self.shuffle
               else torch.arange(self.n, device=self.device))
        for i in range(0, self.n, self.batch_size):
            sel = idx[i:i + self.batch_size]
            yield tuple(t.index_select(0, sel) for t in self.tensors)


class CollocationSampler:
    """Draw M points (x, y, phi) per call, sources read at the nearest mesh cell."""

    def __init__(self, cell_xy):
        xy = cell_xy.detach().cpu().numpy()
        self.tree = cKDTree(xy)
        self.lo = xy.min(axis=0)
        self.hi = xy.max(axis=0)

    def sample(self, M, device):
        xy = np.random.uniform(self.lo, self.hi, size=(M, 2))
        phi = np.random.uniform(0.0, 2.0 * np.pi, M)
        _, cell_idx = self.tree.query(xy)
        y = np.concatenate([xy, phi[:, None]], axis=1)
        return (torch.tensor(y, dtype=torch.float32, device=device),
                torch.as_tensor(cell_idx, dtype=torch.long, device=device))


class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback (Bischof & Kraus, 2021)."""

    def __init__(self, names, alpha=0.999, temperature=0.1, rho_mean=0.999,
                 device="cpu"):
        self.names = list(names)
        self.m = len(self.names)
        self.alpha = alpha
        self.tau = temperature
        self.rho_mean = rho_mean
        self.device = device
        self.lambdas = torch.ones(self.m, device=device)
        self.l0 = None
        self.lprev = None

    def update(self, losses):
        """losses: list of scalars/tensors (one per term, ordered like self.names).
        Returns the (m,) tensor of lambda_i weights (detached)."""
        L = torch.stack([
            l.detach() if isinstance(l, torch.Tensor)
            else torch.as_tensor(float(l), device=self.device)
            for l in losses
        ]).clamp_min(1e-12)

        if self.l0 is None:
            self.l0, self.lprev = L, L
            return self.lambdas

        bal_prev = self.m * torch.softmax(L / (self.tau * self.lprev), dim=0)
        bal_init = self.m * torch.softmax(L / (self.tau * self.l0),   dim=0)
        rho = 1.0 if torch.rand(1).item() < self.rho_mean else 0.0
        self.lambdas = (self.alpha * (rho * self.lambdas + (1.0 - rho) * bal_init)
                        + (1.0 - self.alpha) * bal_prev)
        self.lprev = L
        return self.lambdas


class Trainer:
    def __init__(self, model, device, save_dir=CHECKPOINTS_DIR):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.encoder = None
        self.finetune_encoder = False

    def _encode_branch(self, u1_in, u2_in):
        """Branch-net latents: the raw fields go through the AE inside the graph when
        fine-tuning; otherwise the inputs already are latents."""
        if self.finetune_encoder:
            return (self.encoder.encode_torch("u1_in", u1_in),
                    self.encoder.encode_torch("u2_in", u2_in))
        return u1_in, u2_in

    def _pde_loss(self, pde_batch, y_pde, cell_idx, n_colloc, return_parts=False):
        """Relative physics loss of a PDE batch, normalised as in `utils.rel_phys_loss`.

        n_colloc=0: pde_batch = (enc_u1, enc_u2, u1_eval, u2_eval), the stored pool;
        n_colloc>0: pde_batch = (enc_u1, enc_u2, T_raw, kappa_raw), resampled points."""
        if n_colloc > 0:
            b1, b2, T_raw, kappa_raw = [t.to(self.device) for t in pde_batch]
            T_at, kap_at = T_raw[:, cell_idx], kappa_raw[:, cell_idx]
            u1_eval = kap_at * IB_COEF * T_at ** 4
            u2_eval = -kap_at
        else:
            b1, b2, u1_eval, u2_eval = [t.to(self.device) for t in pde_batch]
        enc_u1, enc_u2 = self._encode_branch(b1, b2)
        residual, I = self.model.pde_residual([enc_u1, enc_u2], y_pde, u1_eval, u2_eval)
        rhs = u1_eval + u2_eval * I
        transport = residual + rhs
        denom = torch.maximum((transport ** 2).mean(),
                              (rhs ** 2).mean()).detach().clamp_min(1e-12)
        mse = torch.mean(residual ** 2)
        if return_parts:
            return mse / denom, mse.detach(), denom
        return mse / denom

    def train(self, dict_pde_train, dict_bc_train, train_dataset,
              dict_pde_val, dict_bc_val, val_dataset,
              epochs=2000, batch_size_pde=64, batch_size_data=8,
              w_phys=1.0, w_bc=100.0, w_data=100.0, save_name="deeponet_final",
              n_colloc=0, colloc_sampler=None,
              loss_balancing="static", relobralo_alpha=0.999,
              relobralo_temp=0.1, relobralo_rho=0.999,
              encoder=None, finetune_encoder=False, unfreeze_encoder_epoch=0,
              encoder_lr=1e-5, base_lr=1e-3, grad_clip=0.0,
              monitor_phys=True, monitor_every=25, resume=True, resume_every=25,
              data_loss="rel_l2"):

        self.encoder = encoder
        self.finetune_encoder = finetune_encoder
        encoder_unfrozen = False
        best_encoder_state = None

        if finetune_encoder:
            optimizer = optim.Adam([
                {"params": list(self.model.parameters()), "lr": base_lr},
                {"params": list(self.encoder.parameters()), "lr": encoder_lr},
            ])
            ue = unfreeze_encoder_epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[lambda e: 0.9977 ** e,
                           lambda e: 0.9977 ** max(0, e - ue)])
            self.encoder.requires_grad_(False)
            self.encoder.eval_mode()
            encoder_unfrozen = unfreeze_encoder_epoch <= 0
            if encoder_unfrozen:
                self.encoder.requires_grad_(True)
                print(f"Encoder fine-tuning ACTIVE from epoch 0 (initial lr={encoder_lr:g}, decaying as gamma^e).")
            else:
                print(f"Encoder fine-tuning: frozen, unfreezing at epoch {unfreeze_encoder_epoch} "
                      f"(initial lr={encoder_lr:g}, decaying from the unfreeze on).")
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9977)
        cell_idx = None
        y_pde = None if n_colloc > 0 else dict_pde_train["y"].to(self.device)
        y_bc = dict_bc_train["y"].to(self.device)

        monitor_phys = monitor_phys and w_phys <= 0
        pde_loader = val_pde_loader = bc_loader = val_bc_loader = None
        if w_phys > 0 or monitor_phys:
            bs_pde = batch_size_pde if w_phys > 0 else min(batch_size_pde, 64)
            if n_colloc > 0:
                pde_loader = DataLoader(TensorDataset(dict_pde_train["u1_in"], dict_pde_train["u2_in"], dict_pde_train["T_raw"], dict_pde_train["kappa_raw"]), batch_size=bs_pde, shuffle=True)
                val_pde_loader = DataLoader(TensorDataset(dict_pde_val["u1_in"], dict_pde_val["u2_in"], dict_pde_val["T_raw"], dict_pde_val["kappa_raw"]), batch_size=bs_pde, shuffle=False)
            else:
                pde_loader = DataLoader(TensorDataset(dict_pde_train["u1_in"], dict_pde_train["u2_in"], dict_pde_train["u1_eval"], dict_pde_train["u2_eval"]), batch_size=bs_pde, shuffle=True)
                val_pde_loader = DataLoader(TensorDataset(dict_pde_val["u1_in"], dict_pde_val["u2_in"], dict_pde_val["u1_eval"], dict_pde_val["u2_eval"]), batch_size=bs_pde, shuffle=False)
        if w_bc > 0:
            bc_loader = DataLoader(TensorDataset(dict_bc_train["u1_in"], dict_bc_train["u2_in"], dict_bc_train["I_target"]), batch_size=batch_size_pde, shuffle=True)
            val_bc_loader = DataLoader(TensorDataset(dict_bc_val["u1_in"], dict_bc_val["u2_in"], dict_bc_val["I_target"]), batch_size=batch_size_pde, shuffle=False)

        data_loader_train = DeviceTensorLoader(
            (train_dataset.u1_in, train_dataset.u2_in, train_dataset.I_target_full),
            batch_size=batch_size_data, shuffle=True, device=self.device)
        data_loader_val = DeviceTensorLoader(
            (val_dataset.u1_in, val_dataset.u2_in, val_dataset.I_target_full),
            batch_size=batch_size_data, shuffle=False, device=self.device)

        loss_phys_list, loss_bc_list, step_list, loss_data_list = [], [], [], []
        val_loss_phys_list, val_loss_bc_list, val_loss_data_list = [], [], []
        loss_phys_abs_list, val_loss_phys_abs_list = [], []
        min_val_loss = float("inf")
        best_model_path = os.path.join(self.save_dir, f"{save_name}.pth")
        y_data_full = train_dataset.y_full.to(self.device)
        scale_sq = self.model.I_scale ** 2
        if data_loss not in ("mse", "rel_l2"):
            raise ValueError(f"unknown data_loss: {data_loss!r} (mse | rel_l2)")

        def data_loss_fn(pred, target):
            """rel_l2: per-snapshot relative L2, the reported metric;
            mse: pooled MSE / I_scale^2."""
            if data_loss == "rel_l2":
                return ((pred - target).norm(dim=1)
                        / target.norm(dim=1).clamp_min(1e-8)).mean()
            return F.mse_loss(pred, target) / scale_sq

        balancer = None
        lambda_history = {}
        if loss_balancing == "relobralo":
            active_names = [n for n, w in (("data", w_data), ("phys", w_phys), ("bc", w_bc)) if w > 0]
            balancer = ReLoBRaLo(active_names, alpha=relobralo_alpha,
                                 temperature=relobralo_temp, rho_mean=relobralo_rho,
                                 device=self.device)
            lambda_history = {n: [] for n in active_names}
            print(f"ReLoBRaLo active on {active_names} "
                  f"(alpha={relobralo_alpha}, tau={relobralo_temp}, E[rho]={relobralo_rho}). "
                  f"It only does something with >= 2 active terms (e.g. w_phys>0).")

        lam_current = balancer.lambdas.clone() if balancer is not None else None

        from utils import TrainingHistory, plot_learning_curves
        hist = TrainingHistory(save_name, save_dir=self.save_dir,
                               meta={"w_data": w_data, "w_phys": w_phys, "w_bc": w_bc,
                                     "epochs": epochs, "loss_balancing": loss_balancing,
                                     "finetune_encoder": bool(finetune_encoder)})
        mon_pde_iter = cycle(pde_loader) if monitor_phys else None
        mon_val_pde_iter = cycle(val_pde_loader) if monitor_phys else None
        if monitor_phys:
            print(f"RTE residual monitoring active (w_phys=0): measured on train+val "
                  f"every {monitor_every} epochs, never optimised.")

        from utils import resume_path, save_atomic
        rpath = resume_path(save_name, "don", self.save_dir) if resume else None
        start_epoch = 0
        if rpath and os.path.exists(rpath):
            st = torch.load(rpath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            min_val_loss, start_epoch = st["min_val_loss"], st["epoch"]
            hist.series = st["hist"]
            torch.set_rng_state(st["rng"].cpu())
            if finetune_encoder and st.get("encoder") is not None:
                for k, m in self.encoder.models.items():
                    m.load_state_dict(st["encoder"][k])
                best_encoder_state = st.get("best_encoder")
            print(f"RESUMING at epoch {start_epoch}/{epochs} (best val = {min_val_loss:.6f})")

        pbar = tqdm(range(start_epoch, epochs), desc="Training", unit="ep",
                    dynamic_ncols=True, mininterval=2.0)
        for i in pbar:
            if finetune_encoder and not encoder_unfrozen and i >= unfreeze_encoder_epoch:
                self.encoder.requires_grad_(True)
                encoder_unfrozen = True
                tqdm.write(f"[Epoch {i}] Encoder UNFROZEN (initial lr={encoder_lr:g}, decaying).")

            if n_colloc > 0:
                y_pde, cell_idx = colloc_sampler.sample(n_colloc, self.device)
            self.model.train()
            if finetune_encoder and encoder_unfrozen:
                self.encoder.train_mode()
            total_loss_epoch, loss_phys_epoch, loss_bc_epoch, loss_data_epoch = 0.0, 0.0, 0.0, 0.0
            loss_phys_abs_epoch = 0.0

            pde_iter = cycle(pde_loader) if w_phys > 0 else None
            bc_iter = cycle(bc_loader) if w_bc > 0 else None

            num_batches = len(data_loader_train)
            for data_batch in data_loader_train:
                u1_in_data, u2_in_data, I_target_data = data_batch

                optimizer.zero_grad()

                b1_data, b2_data = self._encode_branch(u1_in_data, u2_in_data)
                I_pred_data = self.model([b1_data, b2_data], y_data_full)
                loss_data = data_loss_fn(I_pred_data, I_target_data)

                loss_phys = torch.zeros((), device=self.device)
                if w_phys > 0:
                    loss_phys, phys_abs, _ = self._pde_loss(next(pde_iter), y_pde, cell_idx,
                                                            n_colloc, return_parts=True)
                    loss_phys_abs_epoch += phys_abs.item()

                loss_bc = torch.zeros((), device=self.device)
                if w_bc > 0:
                    u1_in_bc, u2_in_bc, I_target_bc = [t.to(self.device) for t in next(bc_iter)]
                    b1_bc, b2_bc = self._encode_branch(u1_in_bc, u2_in_bc)
                    I_pred_bc = self.model([b1_bc, b2_bc], y_bc)
                    loss_bc = torch.mean((I_pred_bc - I_target_bc) ** 2) / scale_sq

                if balancer is not None:
                    active = [(n, l) for n, l, w in (("data", loss_data, w_data),
                                                     ("phys", loss_phys, w_phys),
                                                     ("bc", loss_bc, w_bc)) if w > 0]
                    loss = sum(lam_current[k] * l for k, (_, l) in enumerate(active))
                else:
                    loss = w_phys * loss_phys + w_bc * loss_bc + w_data * loss_data

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], grad_clip)
                optimizer.step()

                loss_phys_epoch += loss_phys.item()
                loss_bc_epoch += loss_bc.item()
                loss_data_epoch += loss_data.item()
                total_loss_epoch += loss.item()

            scheduler.step()

            avg_loss = total_loss_epoch / num_batches
            loss_phys_list.append(loss_phys_epoch / num_batches)
            loss_phys_abs_list.append(loss_phys_abs_epoch / num_batches)
            loss_bc_list.append(loss_bc_epoch / num_batches)
            loss_data_list.append(loss_data_epoch / num_batches)
            step_list.append(i)
            if balancer is not None:
                epoch_losses = {"data": loss_data_epoch / num_batches,
                                "phys": loss_phys_epoch / num_batches,
                                "bc": loss_bc_epoch / num_batches}
                lam_current = balancer.update([epoch_losses[n] for n in balancer.names])
                for k, n in enumerate(balancer.names):
                    lambda_history[n].append(lam_current[k].item())

            self.model.eval()
            if finetune_encoder:
                self.encoder.eval_mode()
            val_total_epoch, val_phys_epoch, val_bc_epoch, val_data_epoch = 0.0, 0.0, 0.0, 0.0
            val_phys_abs_epoch = 0.0
            val_pde_iter = cycle(val_pde_loader) if w_phys > 0 else None
            val_bc_iter = cycle(val_bc_loader) if w_bc > 0 else None
            num_val_batches = len(data_loader_val)

            with torch.no_grad():
                for data_batch in data_loader_val:
                    u1_in_data, u2_in_data, I_target_data_sample = data_batch

                    loss_phys_val = torch.zeros((), device=self.device)
                    if w_phys > 0:
                        with torch.enable_grad():
                            loss_phys_val, phys_abs_val, _ = self._pde_loss(
                                next(val_pde_iter), y_pde, cell_idx, n_colloc, return_parts=True)
                        loss_phys_val = loss_phys_val.detach()
                        val_phys_abs_epoch += phys_abs_val.item()

                    b1_data, b2_data = self._encode_branch(u1_in_data, u2_in_data)
                    I_pred_data_reshaped = self.model([b1_data, b2_data], y_data_full)
                    loss_data_val = data_loss_fn(I_pred_data_reshaped, I_target_data_sample)

                    loss_bc_val = torch.zeros((), device=self.device)
                    if w_bc > 0:
                        u1_in_bc, u2_in_bc, I_target_bc = [t.to(self.device) for t in next(val_bc_iter)]
                        b1_bc, b2_bc = self._encode_branch(u1_in_bc, u2_in_bc)
                        I_pred_bc = self.model([b1_bc, b2_bc], y_bc)
                        loss_bc_val = torch.mean((I_pred_bc - I_target_bc) ** 2) / scale_sq

                    loss_val = w_phys * loss_phys_val + w_bc * loss_bc_val + w_data * loss_data_val

                    val_phys_epoch += loss_phys_val.item()
                    val_bc_epoch += loss_bc_val.item()
                    val_data_epoch += loss_data_val.item()
                    val_total_epoch += loss_val.item()

            phys_train_mon = phys_val_mon = None
            phys_abs_train_mon = phys_abs_val_mon = None
            if monitor_phys and (i % monitor_every == 0 or i == epochs - 1):
                with torch.enable_grad():
                    rel_tr, abs_tr, _ = self._pde_loss(next(mon_pde_iter), y_pde, cell_idx,
                                                       n_colloc, return_parts=True)
                    rel_va, abs_va, _ = self._pde_loss(next(mon_val_pde_iter), y_pde, cell_idx,
                                                       n_colloc, return_parts=True)
                phys_train_mon, phys_abs_train_mon = rel_tr.detach().item(), abs_tr.item()
                phys_val_mon, phys_abs_val_mon = rel_va.detach().item(), abs_va.item()
                del rel_tr, abs_tr, rel_va, abs_va  # no graph kept until the next measurement

            avg_val_loss = (val_data_epoch if balancer is not None else val_total_epoch) / num_val_batches
            val_loss_phys_list.append(val_phys_epoch / num_val_batches)
            val_loss_phys_abs_list.append(val_phys_abs_epoch / num_val_batches)
            val_loss_bc_list.append(val_bc_epoch / num_val_batches)
            val_loss_data_list.append(val_data_epoch / num_val_batches)

            hist.log(i + 1, train_total=avg_loss, train_data=loss_data_list[-1],
                     train_bc=loss_bc_list[-1] if w_bc > 0 else None,
                     train_phys=(loss_phys_list[-1] if w_phys > 0 else phys_train_mon),
                     train_phys_abs=(loss_phys_abs_list[-1] if w_phys > 0 else phys_abs_train_mon),
                     val_data=val_loss_data_list[-1],
                     val_bc=val_loss_bc_list[-1] if w_bc > 0 else None,
                     val_phys=(val_loss_phys_list[-1] if w_phys > 0 else phys_val_mon),
                     val_phys_abs=(val_loss_phys_abs_list[-1] if w_phys > 0 else phys_abs_val_mon))

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                save_atomic(self.model.state_dict(), best_model_path)
                if finetune_encoder:
                    best_encoder_state = {
                        k: {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()}
                        for k, m in self.encoder.models.items()}

            pbar.set_postfix(train=f"{avg_loss:.2f}", val=f"{avg_val_loss:.2f}", best=f"{min_val_loss:.2f}")

            if i % 50 == 0:
                tqdm.write(f"Epoch {i:4d} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best Val: {min_val_loss:.6f}")
                p_tr = loss_phys_list[-1] if w_phys > 0 else phys_train_mon
                p_va = val_loss_phys_list[-1] if w_phys > 0 else phys_val_mon
                suf = "" if w_phys > 0 else " (phys: monitored only)"
                fmt = lambda v: f"{v:.6f}" if v is not None else "  --  "
                tqdm.write(f"           | data={loss_data_list[-1]:.6f} phys={fmt(p_tr)} bc={loss_bc_list[-1]:.6f}"
                           f" | val data={val_loss_data_list[-1]:.6f} phys={fmt(p_va)} bc={val_loss_bc_list[-1]:.6f}{suf}")
                if balancer is not None:
                    tqdm.write("           | lambdas " + "  ".join(
                        f"{n}={lambda_history[n][-1]:.3f}" for n in balancer.names))

            if rpath and (i + 1) % resume_every == 0:
                save_atomic({"epoch": i + 1, "model": self.model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(),
                             "min_val_loss": min_val_loss, "hist": hist.series,
                             "rng": torch.get_rng_state(),
                             "encoder": ({k: m.state_dict() for k, m in self.encoder.models.items()}
                                         if finetune_encoder else None),
                             "best_encoder": best_encoder_state}, rpath)

        if rpath and os.path.exists(rpath):
            os.remove(rpath)
        print(f"Training finished. Best model saved to: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path))
        if finetune_encoder and best_encoder_state is not None:
            for k, m in self.encoder.models.items():
                m.load_state_dict(best_encoder_state[k])
            ft_path = encoder_path(f"{save_name}_ft", self.encoder.kind,
                                   save_dir=self.save_dir)
            self.encoder.save(ft_path)
            print(f"Fine-tuned encoder (best epoch) saved to: {ft_path}")
        hist.meta["best_val"] = min_val_loss
        hist.save()
        try:
            plot_learning_curves(save_name, save_dir=self.save_dir,
                                 title=f"{save_name} — learning curves")
        except Exception as e:
            print(f"  [!] learning-curve plot skipped: {e}")
        if balancer is not None:
            self.plot_lambda_curves(step_list, lambda_history,
                                    save_name=f"{save_name}_lambda_curves.pdf")

    def load_weights(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"Weights loaded from {path}")
        else:
            raise FileNotFoundError(f"Weight file not found: {path}")


    def plot_lambda_curves(self, step_list, lambda_history, save_name="lambda_curves.pdf"):
        """Evolution of the adaptive lambda_i weights (ReLoBRaLo) over the epochs."""
        plt.figure(figsize=(12, 6))
        for name, hist in lambda_history.items():
            plt.plot(step_list, hist, label=f"lambda {name}")
        plt.xlabel("Epochs")
        plt.ylabel("lambda weight (sums to the number of terms)")
        plt.title("ReLoBRaLo adaptive weights")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        chemin_image = save_figure(os.path.join(self.save_dir, save_name))
        print(f"lambda-weight plot saved to: {chemin_image}")
        plt.close()


def build_encoder_from_args(args):
    """Instantiate an unfitted encoder from the CLI hyperparameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return make_encoder(
        args.encoder,
        n_components=args.pca_components,
        use_scaler=not args.pca_no_scaler,
        latent_dim=args.ae_latent_dim,
        epochs=args.ae_epochs,
        batch_size=args.ae_batch_size,
        lr=args.ae_lr,
        device=device,
    )


def run_train_encoder(args):
    if args.name != encoder_run(args.name):
        raise SystemExit(
            f"{args.name!r} names an output head, not an encoder: both heads of "
            f"{encoder_run(args.name)!r} share one.\n"
            f"Fit it under the configuration name: --mode train_encoder --encoder "
            f"{args.encoder} --name {encoder_run(args.name)}")
    print(f"\n--- MODE TRAIN_ENCODER ({args.encoder}) ---")
    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    data_train, data_val = splits["data"]

    fit_dict = {"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]}
    val_dict = {"u1_in": data_val["u1_in"], "u2_in": data_val["u2_in"]}

    encoder = build_encoder_from_args(args)
    encoder.fit_transform_dicts(fit_dict, INPUT_KEYS, dict_val=val_dict)

    path = encoder_path(args.name, args.encoder)
    encoder.save(path)
    print(f"Latent dimensions: {encoder.latent_dims}")
    print(f"[OK] Encoder ready. Now run: --mode train_deeponet --encoder {args.encoder} --name {args.name}")


def _get_or_fit_encoder(args, splits, device):
    """Load the saved encoder. Fallback: fit a PCA on the fly if it is missing."""
    path = encoder_path(args.name, args.encoder)

    if os.path.exists(path):
        print(f"Loading encoder: {path}")
        return load_encoder(args.encoder, path, device=device)

    if args.encoder == "pca":
        print(f"[Fallback] No encoder at {path} -> fitting a PCA on the fly.")
        data_train, _ = splits["data"]
        fit_dict = {"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]}
        encoder = build_encoder_from_args(args)
        encoder.fit_transform_dicts(fit_dict, INPUT_KEYS)
        encoder.save(path)
        return encoder

    raise FileNotFoundError(
        f"Encoder not found: {path}\n"
        f"Run first: python {os.path.basename(__file__)} --mode train_encoder "
        f"--encoder {args.encoder} --name {args.name}"
    )


def run_train_deeponet(args, device):
    print(f"\n--- MODE TRAIN_DEEPONET ({args.encoder}) ---")
    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    dict_pde_train, dict_pde_val = splits["pde"]
    dict_bc_train, dict_bc_val = splits["bc"]
    data_train, data_val = splits["data"]
    y_full = splits["y_full"]

    from_scratch = args.encoder_from_scratch
    if from_scratch:
        encoder = build_encoder_from_args(args)
        if not hasattr(encoder, "init_untrained"):
            raise ValueError(f"--encoder_from_scratch is not supported for '{args.encoder}' "
                             f"(no init_untrained method).")
        print(f"AE FROM SCRATCH: encoder '{args.encoder}' with RANDOM weights "
              f"(latent {args.ae_latent_dim}), trained jointly with the DeepONet "
              f"at the SAME lr ({args.lr:g}, decaying).")
        encoder.init_untrained(
            {"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]}, INPUT_KEYS)
        finetune = True
        unfreeze_epoch = 0
        encoder_lr = args.lr
    else:
        encoder = _get_or_fit_encoder(args, splits, device)
        finetune = args.finetune_encoder
        unfreeze_epoch = args.unfreeze_encoder_epoch
        encoder_lr = args.encoder_lr

    if finetune and encoder.kind != "autoencoder":
        raise ValueError(
            f"--finetune_encoder requires an autoencoder encoder "
            f"(got '{encoder.kind}'). A PCA has no weights to train.")

    if finetune:
        print("Encoder fine-tuning: on-the-fly encoding (raw fields kept).")
        train_dataset = FullGridDataset(data_train["u1_in"], data_train["u2_in"], data_train["I_target"], y_full)
        val_dataset = FullGridDataset(data_val["u1_in"], data_val["u2_in"], data_val["I_target"], y_full)
    else:
        dict_pde_train = encoder.transform_dicts(dict_pde_train, INPUT_KEYS)
        dict_pde_val = encoder.transform_dicts(dict_pde_val, INPUT_KEYS)
        dict_bc_train = encoder.transform_dicts(dict_bc_train, INPUT_KEYS)
        dict_bc_val = encoder.transform_dicts(dict_bc_val, INPUT_KEYS)

        enc_data_train = encoder.transform_dicts({"u1_in": data_train["u1_in"], "u2_in": data_train["u2_in"]}, INPUT_KEYS)
        enc_data_val = encoder.transform_dicts({"u1_in": data_val["u1_in"], "u2_in": data_val["u2_in"]}, INPUT_KEYS)
        train_dataset = FullGridDataset(enc_data_train["u1_in"], enc_data_train["u2_in"], data_train["I_target"], y_full)
        val_dataset = FullGridDataset(enc_data_val["u1_in"], enc_data_val["u2_in"], data_val["I_target"], y_full)

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

    print(f"FullGridDataset built: {len(train_dataset)} train snapshots, {len(val_dataset)} val.")

    branch_input_dims = [encoder.latent_dims["u1_in"], encoder.latent_dims["u2_in"]]
    print(f"Branch-net latent dimensions: {branch_input_dims}")

    I_target_train = data_train["I_target"]
    I_non_zero = I_target_train[I_target_train > 1e-3]
    median_I = torch.median(I_non_zero).item() if len(I_non_zero) > 0 else 1.0
    print(f"Computed I_scale (median): {median_I:.4e}")

    model = MultiBranchDeepONet(
        branch_input_dims=branch_input_dims,
        trunk_input_dim=QUERY_DIM,
        p_basis=args.p_basis,
        trunk_hidden=[512, 512, 512, 512],
        I_scale_val=median_I,
        trunk_activate_last=args.trunk_activate_last,
        affine=args.affine,
    ).to(device)
    print("Head: I = (<beta, gamma(y)> + b) * I_scale -- AFFINE reconstructor (Thm 3.4)"
          if args.affine else "Head: I = softplus(<beta, gamma(y)> + b) * I_scale")

    trainer = Trainer(model, device, save_dir=CHECKPOINTS_DIR)
    trainer.train(
        dict_pde_train, dict_bc_train, train_dataset,
        dict_pde_val, dict_bc_val, val_dataset,
        epochs=args.epochs,
        batch_size_pde=args.batch_size_pde,
        batch_size_data=args.batch_size_data,
        w_phys=args.w_phys,
        w_bc=args.w_bc,
        w_data=args.w_data,
        save_name=args.name,
        n_colloc=args.n_colloc,
        colloc_sampler=colloc_sampler,
        loss_balancing=args.loss_balancing,
        relobralo_alpha=args.relobralo_alpha,
        relobralo_temp=args.relobralo_temp,
        relobralo_rho=args.relobralo_rho,
        encoder=encoder,
        finetune_encoder=finetune,
        unfreeze_encoder_epoch=unfreeze_epoch,
        encoder_lr=encoder_lr,
        base_lr=args.lr,
        grad_clip=args.grad_clip,
        data_loss=args.data_loss,
        monitor_phys=args.monitor_phys,
        monitor_every=args.monitor_every,
        resume=args.resume,
    )


def run_eval_2d(args, device):
    print(f"\n--- MODE EVAL_2D ({args.encoder}) ---")

    enc_path = encoder_path(args.name, args.encoder)
    ft_path = encoder_path(f"{args.name}_ft", args.encoder)
    if os.path.exists(ft_path):
        print(f"Fine-tuned encoder detected: {ft_path} (used instead of the original).")
        encoder = load_encoder(args.encoder, ft_path, device=device)
    elif os.path.exists(enc_path):
        encoder = load_encoder(args.encoder, enc_path, device=device)
    else:
        raise FileNotFoundError(f"Encoder not found: {enc_path}. Run train_encoder first.")
    branch_input_dims = [encoder.latent_dims["u1_in"], encoder.latent_dims["u2_in"]]
    print(f"Encoder loaded. Latent dimensions: {branch_input_dims}")

    model_path = args.model_path or os.path.join(CHECKPOINTS_DIR, f"{args.name}.pth")
    model = load_model(model_path, device)

    h5 = solver_h5()
    has_solver_h5 = h5 != MESH_STUB
    evaluator = Evaluator(model, device, h5_path=h5)
    evaluator.encoder = encoder

    splits = load_and_split_dataset(args.dataset, map_location="cpu")
    data_eval = split_data(splits, args.eval_split)
    y_full = splits["y_full"]

    print(f"\n--- Comparison maps (integration of I) "
          f"-- {args.eval_split.upper()} split ---")
    snaps = resolve_snapshots(args.eval_snapshots, data_eval)
    plot_eval_snapshots(evaluator, data_eval, y_full, spec=snaps,
                        output_dir=args.output_dir)
    if has_solver_h5:
        evaluator.plot_snapshot_comparison(snapshot_idx=0, output_dir=args.output_dir)
        evaluator.plot_snapshot_comparison(snapshot_idx=20, output_dir=args.output_dir)

    print(f"\n--- Errors on the intensity I ({args.eval_split.upper()} split) ---")

    enc_eval = encoder.transform_dicts(
        {"u1_in": data_eval["u1_in"], "u2_in": data_eval["u2_in"]}, INPUT_KEYS)
    eval_loader = DataLoader(
        TensorDataset(enc_eval["u1_in"], enc_eval["u2_in"], data_eval["I_target"]),
        batch_size=args.batch_size_data,
        shuffle=False,
    )
    g_ref = None
    if getattr(args, "g_ref", "integrated") == "solver":
        from utils import solver_G
        g_ref = solver_G(splits[f"{args.eval_split}_idx"])
        print(f"   G reference: the solver's own G field ({tuple(g_ref.shape)})")
    return evaluator.evaluate_split_errors(eval_loader, y_coords=y_full,
                                           label=args.eval_split.upper(), g_ref=g_ref)


def build_parser():
    parser = argparse.ArgumentParser(description="Modular DeepONet for the 2D RTE (PCA or autoencoder)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train_encoder", "train_deeponet", "eval_2d"],
                        help="Pipeline stage to run")
    parser.add_argument("--encoder", type=str, default="pca",
                        choices=["pca", "autoencoder"],
                        help="Branch-net input encoder (default: pca). The "
                             "Transformer-Based DeepONet encoder is JOINT over the two "
                             "channels: see deeponet_joint.py --joint_encoder tbd.")
    parser.add_argument("--name", type=str, default="deeponet_rte2d", help="Experiment name")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to the .pt dataset")

    parser.add_argument("--epochs", type=int, default=2000, help="DeepONet training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial DeepONet learning rate (default 1e-3, decayed by "
                             "gamma=0.9977). With --encoder_from_scratch the AE shares it.")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Gradient-norm clipping (0 = off). A value > 0 (e.g. 1.0) "
                             "prevents the I=0 collapse under aggressive fine-tuning "
                             "(high encoder lr).")
    parser.add_argument("--batch_size_pde", type=int, default=2024, help="PDE/BC batch size")
    parser.add_argument("--batch_size_data", type=int, default=256, help="Data batch size")
    parser.add_argument("--w_phys", type=float, default=0.0, help="Weight of the PDE loss")
    parser.add_argument("--w_bc", type=float, default=0.0, help="Weight of the BC loss")
    parser.add_argument("--w_data", type=float, default=1.0, help="Weight of the data loss")
    parser.add_argument("--n_colloc", type=int, default=50000,
                        help="Number of PDE collocation points resampled every epoch "
                             "(0 = the pool stored in the dataset).")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume an interrupted run from "
                             "checkpoints/<name>_don_resume.pt (default). --no-resume to "
                             "start over, ignoring the saved state.")
    parser.add_argument("--monitor_phys", action=argparse.BooleanOptionalAction, default=True,
                        help="Measure the RTE residual (train+val) every --monitor_every "
                             "epochs EVEN in data-only mode (w_phys=0), so its curve can be "
                             "plotted. No effect when w_phys>0 (the real loss is already "
                             "recorded). --no-monitor_phys to switch it off.")
    parser.add_argument("--monitor_every", type=int, default=25,
                        help="Period, in epochs, of the RTE residual monitoring.")
    parser.add_argument("--data_loss", type=str, default="rel_l2", choices=["rel_l2", "mse"],
                        help="Data loss: rel_l2 = per-snapshot relative L2 over the full "
                             "field (Lu et al.; the default here, and the reported metric); "
                             "mse = pooled MSE / I_scale^2")
    parser.add_argument("--p_basis", type=int, default=128, help="Size of the latent basis (trunk/branch)")
    parser.add_argument("--trunk_activate_last", action="store_true",
                        help="Activate (GELU) the last trunk layer, as the RTENet code of "
                             "Lu et al. 2026 does (default: linear last layer)")

    parser.add_argument("--finetune_encoder", action="store_true",
                        help="Train (unfreeze) the autoencoder weights while training the "
                             "DeepONet: the raw fields are encoded on the fly, inside the "
                             "graph. Default: FROZEN encoder (precomputed latents). "
                             "Incompatible with --encoder pca.")
    parser.add_argument("--unfreeze_encoder_epoch", type=int, default=0,
                        help="Epoch at which to unfreeze the encoder (0 = from the start). "
                             "Before it the AE is frozen. Only effective together with "
                             "--finetune_encoder.")
    parser.add_argument("--encoder_lr", type=float, default=1e-5,
                        help="Initial learning rate of the encoder weights while fine-tuning "
                             "(default 1e-5). Decays as gamma^(e-unfreeze) (gamma=0.9977) FROM "
                             "the unfreeze on: the encoder starts at this value when it "
                             "becomes trainable, then decays like the DeepONet.")
    parser.add_argument("--encoder_from_scratch", action="store_true",
                        help="Do NOT use a pretrained encoder: instantiate an AE with RANDOM "
                             "weights and train it jointly with the DeepONet (from scratch), "
                             "at the SAME learning rate (1e-3, decaying). Forces "
                             "--finetune_encoder with an epoch-0 unfreeze; ignores "
                             "--encoder_lr and --unfreeze_encoder_epoch. Needs an encoder "
                             "with weights (autoencoder).")

    parser.add_argument("--loss_balancing", type=str, default="static",
                        choices=["static", "relobralo"],
                        help="Loss weighting: 'static' (fixed w_*) or 'relobralo' "
                             "(adaptive; the w_* then act as on/off switches)")
    parser.add_argument("--relobralo_alpha", type=float, default=0.99,
                        help="ReLoBRaLo: EMA rate alpha (weight memory, ~1/(1-alpha) steps)")
    parser.add_argument("--relobralo_temp", type=float, default=0.1,
                        help="ReLoBRaLo: softmax temperature tau (small = sharp contrasts)")
    parser.add_argument("--relobralo_rho", type=float, default=0.999,
                        help="ReLoBRaLo: Bernoulli E[rho]; probability (1-rho) of a lookback to t=0")

    parser.add_argument("--affine", action="store_true",
                        help="Drop the output softplus: the reconstructor is then affine "
                             "in beta, exactly the R of Theorem 3.4")
    parser.add_argument("--pca_components", type=int, default=500,
                        help="Number of PCA axes kept per input field")
    parser.add_argument("--pca_no_scaler", action="store_true",
                        help="Raw PCA without StandardScaler (= POD, the convention of Lu et al. 2026)")

    parser.add_argument("--ae_latent_dim", type=int, default=500, help="Autoencoder latent size")
    parser.add_argument("--ae_epochs", type=int, default=600, help="Autoencoder training epochs")
    parser.add_argument("--ae_batch_size", type=int, default=64, help="Autoencoder batch size")
    parser.add_argument("--ae_lr", type=float, default=1e-3, help="Autoencoder learning rate")

    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the .pth (default: checkpoints/<name>.pth)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Evaluation output directory (default: results/<name>/)")
    parser.add_argument("--g_ref", type=str, default="integrated",
        choices=("integrated", "solver"),
        help="Reference for G: the integrated sum_m I_target (default), or the solver's own G field read from RTE2D_SOLVER_H5.")
    parser.add_argument("--eval_split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Split eval_2d measures and plots on. TEST by default: the val "
                             "split was used to pick the checkpoint, so reporting it would "
                             "flatter the model.")
    parser.add_argument("--eval_snapshots", type=str, default=SNAPSHOT_SPEC_DEFAULT,
                        help="Snapshots of the eval split for the comparison maps, comma "
                             "separated: direct index ('0,20'), -1 = the most intense one, "
                             "'structure[:k]' = the k most structured, 'chaotic[:k]' = the k of "
                             "largest spatial variance of G, 'random[:k]' = k drawn at random "
                             "(fixed seed, hence the same snapshots for every model). "
                             f"Default: '{SNAPSHOT_SPEC_DEFAULT}' = 8 snapshots.")
    return parser


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
