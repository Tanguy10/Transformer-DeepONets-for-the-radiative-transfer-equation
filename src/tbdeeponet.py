import contextlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, TensorDataset


def valid_subsample_factors(grid_shape, patch, requested=None, min_tokens=2):
    """Subsampling factors s compatible with the grid: H/s and W/s integers divisible
    by `patch`, at least `min_tokens` tokens per side. `requested=None`: all of them."""
    H, W = grid_shape
    cand = requested if requested is not None else range(1, min(H, W) // patch + 1)
    out = []
    for s in cand:
        s = int(s)
        if s < 1 or H % s or W % s:
            continue
        h, w = H // s, W // s
        if h % patch or w % patch:
            continue
        if (h // patch) < min_tokens or (w // patch) < min_tokens:
            continue
        out.append(s)
    return tuple(out) or (1,)


def parse_subsample_arg(value):
    """Turn a CLI argument into `subsample_factors`:
    'auto' -> None (auto-computed); 'none'/'off'/'1' -> (1,) (full resolution only);
    '1,3,5' -> (1, 3, 5) (later filtered down to the valid factors)."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v == "auto":
        return None
    if v in ("none", "off"):
        return (1,)
    return tuple(int(x) for x in v.replace(" ", "").split(",") if x)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention block: x + MSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: q + MHA(LN(q), LN(kv), LN(kv)), then q + MLP(LN(q))."""

    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, q, kv):
        qn, kvn = self.norm_q(q), self.norm_kv(kv)
        q = q + self.attn(qn, kvn, kvn, need_weights=False)[0]
        q = q + self.mlp(self.norm2(q))
        return q


class RandomFourierEmbedding(nn.Module):
    """Random Fourier features [sin(2pi Bc), cos(2pi Bc)] -> Linear, B fixed of scale sigma."""

    def __init__(self, in_dim, dim, n_freq=64, sigma=3.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freq) * sigma)
        self.proj = nn.Linear(2 * n_freq, dim)

    def forward(self, coords):
        proj = 2 * math.pi * (coords @ self.B)
        feats = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return self.proj(feats)


class FunctionAutoencoder(nn.Module):
    """Patch/Perceiver encoder and coordinate decoder: (N, C, H, W) -> latents -> values.

    encode(x) -> (N, n_latent*dim); encode_tokens(x) -> (N, n_latent, dim);
    decode(z, coords) -> (N, M, C), or (N, M) when C == 1."""

    def __init__(self, grid_shape=(150, 150), patch=10, dim=128, n_latent=4,
                 depth=4, dec_depth=4, heads=8, mlp_ratio=2.0,
                 n_fourier=64, fourier_sigma=3.0, in_channels=2):
        super().__init__()
        H, W = grid_shape
        if H % patch or W % patch:
            raise ValueError(f"patch={patch} must divide the grid {grid_shape}")
        self.grid_shape = (H, W)
        self.patch = patch
        self.dim = dim
        self.n_latent = n_latent
        self.in_channels = in_channels
        self.gh, self.gw = H // patch, W // patch
        self.n_tokens = self.gh * self.gw
        self.latent_dim = n_latent * dim

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.pe_enc = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        self.latent_queries = nn.Parameter(torch.zeros(1, n_latent, dim))
        self.perceiver = CrossAttentionBlock(dim, heads, mlp_ratio)                    
        self.enc_blocks = nn.ModuleList(                                                  
            TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth))
        self.enc_norm = nn.LayerNorm(dim)

        self.coord_embed = RandomFourierEmbedding(2, dim, n_fourier, fourier_sigma)
        self.dec_blocks = nn.ModuleList(
            CrossAttentionBlock(dim, heads, mlp_ratio) for _ in range(dec_depth))
        self.dec_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, in_channels))

        nn.init.trunc_normal_(self.pe_enc, std=0.02)
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

    def _interp_pe(self, h, w):
        """Positional embeddings of the reference grid, bicubic-interpolated to (h, w)."""
        if h == self.gh and w == self.gw:
            return self.pe_enc
        pe = self.pe_enc.reshape(1, self.gh, self.gw, self.dim).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, h * w, self.dim)

    def encode_tokens(self, x):
        """(N, C, H, W) -> latents (N, n_latent, dim), fixed size."""
        p = self.patch_embed(x)
        h, w = p.shape[-2], p.shape[-1]
        f_pe = p.flatten(2).transpose(1, 2)
        f_pe = f_pe + self._interp_pe(h, w)
        z = self.latent_queries.expand(x.shape[0], -1, -1)
        z = self.perceiver(z, f_pe)
        z = self.enc_norm(z)
        for blk in self.enc_blocks:
            z = blk(z)
        return z

    def encode(self, x):
        """(N, C, H, W) -> flattened latent (N, n_latent*dim)."""
        return self.encode_tokens(x).flatten(1)

    def decode(self, z_tokens, coords):
        """Latents (N, n_latent, dim) and coords (M, 2) or (N, M, 2)
        -> (N, M, C), or (N, M) when C == 1."""
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(z_tokens.shape[0], -1, -1)
        x = self.coord_embed(coords)
        for blk in self.dec_blocks:
            x = blk(x, z_tokens)
        x = self.dec_norm(x)
        out = self.head(x)
        return out.squeeze(-1) if self.in_channels == 1 else out

    def forward(self, x, coords):
        z = self.encode_tokens(x)
        return self.decode(z, coords), z.flatten(1)


class TBDEncoder:
    """Joint 2-channel Transformer-Based DeepONet encoder, standardised per channel.

    `transform_dicts` returns the flattened latent under BOTH u1_in and u2_in."""

    kind = "tbd"

    def __init__(self, latent_dim=None, grid_shape=(150, 150), epochs=600, batch_size=64,
                 lr=1e-3, device=None, encode_batch=256,
                 patch=10, dim=128, n_latent=4, depth=4, dec_depth=4, heads=8,
                 mlp_ratio=2.0, n_fourier=64, fourier_sigma=3.0, n_query=4096,
                 subsample_factors=None):
        self.grid_shape = tuple(grid_shape)
        self.patch = patch
        self.dim = dim
        self.n_latent = n_latent
        self.depth = depth
        self.dec_depth = dec_depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.n_fourier = n_fourier
        self.fourier_sigma = fourier_sigma
        self.n_query = n_query
        self.subsample_factors = subsample_factors
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encode_batch = encode_batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_dim = n_latent * dim
        self.model = None
        self.stats = None
        self.latent_dims = {"u1_in": self.latent_dim, "u2_in": self.latent_dim}
        self.fitted = False
        self.coords = self._grid_coords(grid_shape)

    @staticmethod
    def _grid_coords(grid_shape):
        """Cell coordinates in [0, 1]^2, in the flattening order of the fields."""
        H, W = grid_shape
        ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        x = jj.reshape(-1).float() / (W - 1)
        y = ii.reshape(-1).float() / (H - 1)
        return torch.stack([x, y], dim=1)

    def _stack(self, u1, u2):
        """(N, N_cells)×2 -> (N, 2, H, W)."""
        H, W = self.grid_shape
        return torch.cat([u1.float().view(-1, 1, H, W), u2.float().view(-1, 1, H, W)], dim=1)

    def _standardize(self, data):
        mean, std = self.stats
        return (data - mean.to(data.device)) / std.to(data.device)

    def _make_model(self):
        return FunctionAutoencoder(
            grid_shape=self.grid_shape, patch=self.patch, dim=self.dim, n_latent=self.n_latent,
            depth=self.depth, dec_depth=self.dec_depth, heads=self.heads, mlp_ratio=self.mlp_ratio,
            n_fourier=self.n_fourier, fourier_sigma=self.fourier_sigma, in_channels=2)

    def fit_transform_dicts(self, dict_train, input_keys=("u1_in", "u2_in"), dict_val=None,
                            resume_name=None, resume_every=10):
        """Train on the fields; `resume_name` enables restarting after an interruption."""
        coords = self.coords.to(self.device)
        n_cells = coords.shape[0]
        factors = valid_subsample_factors(self.grid_shape, self.patch, self.subsample_factors)
        print(f"[joint Transformer-Based DeepONet] Random subsampling during training: "
              f"factors {factors}")
        gen = torch.Generator().manual_seed(0)
        val_idx = torch.randint(0, n_cells, (min(2 * self.n_query, n_cells),), generator=gen).to(self.device)
        coords_val = coords[val_idx]

        data = self._stack(dict_train["u1_in"], dict_train["u2_in"])
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
        self.stats = (mean, std)
        data_scaled = self._standardize(data)

        val_loader = None
        if dict_val is not None and "u1_in" in dict_val and "u2_in" in dict_val:
            dv = self._standardize(self._stack(dict_val["u1_in"], dict_val["u2_in"]))
            val_loader = DataLoader(TensorDataset(dv), batch_size=self.batch_size, shuffle=False)

        model = self._make_model().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loader = DataLoader(TensorDataset(data_scaled), batch_size=self.batch_size, shuffle=True)

        print(f"\n[joint Transformer-Based DeepONet] Training (n_latent={self.n_latent}, "
              f"dim={self.dim}, "
              f"latent d={self.latent_dim}, 2 channels, M={self.n_query} coords) on {self.device} ...")
        best_val, best_state = float("inf"), None

        from utils import resume_path, save_atomic
        rpath = resume_path(resume_name, "enc") if resume_name else None
        start_epoch = 0
        if rpath and os.path.exists(rpath):
            st = torch.load(rpath, map_location=self.device, weights_only=False)
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            best_val, best_state = st["best_val"], st["best_state"]
            start_epoch = st["epoch"]
            torch.set_rng_state(st["rng"].cpu())
            self.stats = tuple(t.to(self.device) for t in st["stats"])
            print(f"  [joint Transformer-Based DeepONet] RESUMING at epoch "
                  f"{start_epoch}/{self.epochs} "
                  f"(best val = {best_val:.6f})")

        for epoch in range(start_epoch, self.epochs):
            model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                s = factors[torch.randint(len(factors), (1,)).item()]
                enc_in = batch[:, :, ::s, ::s] if s > 1 else batch
                idx = torch.randint(0, n_cells, (self.n_query,), device=self.device)
                target = batch.view(batch.shape[0], 2, -1)[:, :, idx].permute(0, 2, 1)
                optimizer.zero_grad()
                pred, _ = model(enc_in, coords[idx])
                loss = F.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_mse = epoch_loss / len(loader)

            val_mse = None
            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    v = 0.0
                    for (vb,) in val_loader:
                        vb = vb.to(self.device)
                        tgt = vb.view(vb.shape[0], 2, -1)[:, :, val_idx].permute(0, 2, 1)
                        pv, _ = model(vb, coords_val)
                        v += F.mse_loss(pv, tgt).item()
                val_mse = v / len(val_loader)
                if val_mse < best_val:
                    best_val = val_mse
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = (f"  [joint Transformer-Based DeepONet] Epoch {epoch+1:3d}/{self.epochs}"
                       f" | Train MSE: {train_mse:.6f}")
                if val_mse is not None:
                    msg += f" | Val MSE: {val_mse:.6f} | Best Val: {best_val:.6f}"
                print(msg)

            if rpath and (epoch + 1) % resume_every == 0:
                save_atomic({"epoch": epoch + 1, "model": model.state_dict(),
                             "optimizer": optimizer.state_dict(), "best_val": best_val,
                             "best_state": best_state, "rng": torch.get_rng_state(),
                             "stats": tuple(t.cpu() for t in self.stats)}, rpath)

        if rpath and os.path.exists(rpath):
            os.remove(rpath)

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"  [joint Transformer-Based DeepONet] Best model reloaded "
                  f"(Val MSE = {best_val:.6f})")
        model.eval()
        self.model = model
        self.fitted = True
        return self.transform_dicts(dict_train, input_keys)

    def transform_dicts(self, dict_in, input_keys=("u1_in", "u2_in")):
        if not self.fitted:
            raise ValueError("TBDEncoder must be fitted (train_encoder) first.")
        z = self._encode(dict_in["u1_in"], dict_in["u2_in"])
        out = dict_in.copy()
        out["u1_in"] = z
        out["u2_in"] = z
        return out

    def _encode(self, u1, u2):
        self.model.eval()
        data = self._standardize(self._stack(u1, u2))
        outs = []
        with torch.no_grad():
            for i in range(0, data.shape[0], self.encode_batch):
                outs.append(self.model.encode(data[i:i + self.encode_batch].to(self.device)).cpu())
        return torch.cat(outs, dim=0).float()

    def reconstruct_fields(self, u1, u2):
        """(T_hat, kappa_hat) at physical scale, (N, N_cells) each."""
        self.model.eval()
        coords = self.coords.to(self.device)
        data = self._standardize(self._stack(u1, u2))
        outs = []
        with torch.no_grad():
            for i in range(0, data.shape[0], self.encode_batch):
                z = self.model.encode_tokens(data[i:i + self.encode_batch].to(self.device))
                preds = [self.model.decode(z, coords[c0:c0 + self.n_query]).cpu()
                         for c0 in range(0, coords.shape[0], self.n_query)]
                outs.append(torch.cat(preds, dim=1))
        rec = torch.cat(outs, dim=0)
        mean, std = self.stats
        rec = rec * std.reshape(-1).cpu() + mean.reshape(-1).cpu()
        n = rec.shape[0]
        return rec[..., 0].reshape(n, -1).float(), rec[..., 1].reshape(n, -1).float()

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mean, std = self.stats
        state = {
            "kind": self.kind, "latent_dim": self.latent_dim, "grid_shape": self.grid_shape,
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
            "latent_dims": self.latent_dims, "model": self.model.state_dict(),
            "stats": (mean.cpu(), std.cpu()),
            "arch": {"patch": self.patch, "dim": self.dim, "n_latent": self.n_latent,
                     "depth": self.depth, "dec_depth": self.dec_depth, "heads": self.heads,
                     "mlp_ratio": self.mlp_ratio, "n_fourier": self.n_fourier,
                     "fourier_sigma": self.fourier_sigma, "n_query": self.n_query,
                     "subsample_factors": self.subsample_factors},
        }
        torch.save(state, path)
        print(f"Joint Transformer-Based DeepONet encoder saved to: {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device)
        enc = cls(grid_shape=state["grid_shape"], epochs=state.get("epochs", 600),
                  batch_size=state.get("batch_size", 64), lr=state.get("lr", 1e-3),
                  device=device, **state.get("arch", {}))
        enc.latent_dims = state["latent_dims"]
        enc.model = enc._make_model().to(device)
        enc.model.load_state_dict(state["model"])
        enc.model.eval()
        mean, std = state["stats"]
        enc.stats = (mean.to(device), std.to(device))
        enc.fitted = True
        return enc


class CoordDecoder(nn.Module):
    def __init__(self, dim=128, dec_depth=4, heads=8, mlp_ratio=2.0,
                 n_fourier=64, fourier_sigma=3.0, coord_dim=3, out_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.coord_embed = RandomFourierEmbedding(coord_dim, dim, n_fourier, fourier_sigma)
        self.dec_blocks = nn.ModuleList(
            CrossAttentionBlock(dim, heads, mlp_ratio) for _ in range(dec_depth))
        self.dec_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, out_channels))

    def forward(self, z_tokens, coords):
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(z_tokens.shape[0], -1, -1)
        x = self.coord_embed(coords)
        for blk in self.dec_blocks:
            x = blk(x, z_tokens)
        x = self.dec_norm(x)
        out = self.head(x)
        return out.squeeze(-1) if self.out_channels == 1 else out


class TBDOperator(nn.Module):
    def __init__(self, ae, I_scale=1.0, dec_depth=None, heads=8, mlp_ratio=2.0,
                 n_fourier=64, fourier_sigma=3.0, affine=False):
        super().__init__()
        self.ae = ae
        self.dim = ae.dim
        self.n_latent = ae.n_latent
        self.affine = affine
        dec_depth = dec_depth if dec_depth is not None else len(ae.dec_blocks)
        self.dec_I = CoordDecoder(
            dim=ae.dim, dec_depth=dec_depth, heads=heads, mlp_ratio=mlp_ratio,
            n_fourier=n_fourier, fourier_sigma=fourier_sigma, coord_dim=3, out_channels=1)
        self.register_buffer("I_scale", torch.tensor(float(I_scale), dtype=torch.float32))
        self.register_buffer("xy_lo", torch.zeros(2))
        self.register_buffer("xy_hi", torch.ones(2))

    def set_coord_norm(self, y_full):
        """Set the [0,1] normalisation of the (x, y) coords from all the RTE coords."""
        xy = y_full[:, :2]
        self.xy_lo.copy_(xy.min(dim=0).values.to(self.xy_lo.device))
        self.xy_hi.copy_(xy.max(dim=0).values.to(self.xy_hi.device))

    def _norm_coords(self, coords):
        """(x, y, phi) -> [0, 1]^3, the input of the random Fourier features."""
        span = (self.xy_hi - self.xy_lo).clamp_min(1e-8)
        xy = (coords[..., :2] - self.xy_lo) / span
        phi = coords[..., 2:3] / (2 * math.pi)
        return torch.cat([xy, phi], dim=-1)

    def encode_tokens(self, x):
        return self.ae.encode_tokens(x)

    def predict_I(self, z_tokens, coords):
        """(z_tokens, coords (x, y, phi)) -> softplus(D_I) * I_scale, (N, M).

        `affine` drops the softplus."""
        raw = self.dec_I(z_tokens, self._norm_coords(coords))
        return (raw if self.affine else F.softplus(raw)) * self.I_scale

    def reconstruct(self, z_tokens, coords2d):
        """Reconstruction of the input pair (2 channels), for the auxiliary loss."""
        return self.ae.decode(z_tokens, coords2d)

    def forward(self, branch_inputs, coords):
        z_flat = branch_inputs[0]
        z_tokens = z_flat.view(z_flat.shape[0], self.n_latent, self.dim)
        return self.predict_I(z_tokens, coords)

    def pde_residual(self, branch_inputs, y, u1_eval, u2_eval, create_graph=True):
        """RTE residual by autodiff through dec_I (double backward).

        branch_inputs[0]: flattened latent (B, n_latent*dim); y: (P, 3), shared;
        u1_eval, u2_eval: (B, P). Returns (residual, I), both (B, P).
        `create_graph=False` measures the residual without optimising it."""
        from utils import transport_coeffs

        z_flat = branch_inputs[0]
        B = z_flat.shape[0]
        z_tokens = z_flat.view(B, self.n_latent, self.dim)
        backend = sdpa_kernel(SDPBackend.MATH) if create_graph else contextlib.nullcontext()
        with torch.enable_grad(), backend:
            # clone(): a broadcast leaf would make autograd sum dI/dx over the batch.
            xy = y[:, :2].unsqueeze(0).expand(B, -1, -1).clone().requires_grad_(True)
            phi = y[:, 2:3].unsqueeze(0).expand(B, -1, -1)
            I = self.predict_I(z_tokens, torch.cat([xy, phi], dim=-1))
            grad_xy, = torch.autograd.grad(I.sum(), xy, create_graph=create_graph,
                                           retain_graph=create_graph)
        dI_dx, dI_dy = grad_xy[..., 0], grad_xy[..., 1]
        sin_phi, cos_phi = transport_coeffs(phi[..., 0])
        transport = sin_phi * dI_dx + cos_phi * dI_dy
        residual = transport - (u1_eval + u2_eval * I)
        return residual, I

    def init_dec_I_from_recon(self):
        """Warm start: copy the cross-attention blocks of the recon decoder into dec_I."""
        self.dec_I.dec_blocks.load_state_dict(self.ae.dec_blocks.state_dict())
        self.dec_I.dec_norm.load_state_dict(self.ae.dec_norm.state_dict())
        print("[operator] dec_I warm-started from the reconstruction decoder.")


def _encode_all_tokens(operator, fields, batch, device):
    """Encode every field (N,2,H,W) -> tokens (N, n_latent, dim), in chunks, under no_grad."""
    operator.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, fields.shape[0], batch):
            outs.append(operator.encode_tokens(fields[i:i + batch].to(device)).cpu())
    return torch.cat(outs, dim=0)


def fit_operator(operator, encoder, dict_train, dict_val, y_full, I_train, I_val,
                 epochs=1000, batch_size=64, lr=1e-3, n_query=8192, w_recon=0.0,
                 freeze_encoder=True, device="cuda", log_every=10, encode_batch=256,
                 val_every=25, encoder_lr=None, save_path=None,
                 w_phys=0.0, w_bc=0.0, w_data=1.0, n_query_phys=None,
                 loss_balancing="static", relobralo_alpha=0.99, relobralo_temp=0.1,
                 relobralo_rho=0.999, monitor_phys=True, history_name=None, resume=True,
                 data_loss="rel_l2"):
    """Train the (T, kappa) -> I operator on `n_query` random points per step.

    The encoder is frozen (latents precomputed) or fine-tuned inside the graph;
    `w_phys` > 0 adds the relative RTE residual; `data_loss` is rel_l2 or mse."""
    if w_bc > 0:
        raise NotImplementedError(
            "w_bc > 0 is not wired for train_operator: the domain boundary condition "
            "(walls at a uniform 300 K) is not validated with certainty yet -- keep "
            "w_bc=0 for now.")
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    H, W = encoder.grid_shape
    P = y_full.shape[0]
    n_cells = H * W
    coords_rte = y_full.float().to(device)
    coords_2d = encoder.coords.to(device)
    scale_sq = float(operator.I_scale) ** 2
    if data_loss not in ("mse", "rel_l2"):
        raise ValueError(f"data_loss inconnu : {data_loss!r} (mse | rel_l2)")
    rel_l2 = data_loss == "rel_l2"

    def data_loss_fn(pred, target):
        if rel_l2:
            return ((pred - target).norm(dim=1) / target.norm(dim=1).clamp_min(1e-8)).mean()
        return F.mse_loss(pred, target) / scale_sq

    from utils import TrainingHistory, rel_phys_loss
    phys_active = w_phys > 0
    balancer = lam_current = None
    hist = TrainingHistory(history_name or "operator",
                           meta={"w_data": w_data, "w_phys": w_phys, "epochs": epochs,
                                 "loss_balancing": loss_balancing,
                                 "freeze_encoder": bool(freeze_encoder)})

    if phys_active or monitor_phys:
        from deeponet import IB_COEF
        n_dirs_data = int(torch.unique(coords_rte[:, 2]).shape[0])
        cell_idx_pde = torch.arange(P, device=device) // n_dirs_data
        T_train_raw = dict_train["u1_in"].float().to(device)
        kappa_train_raw = dict_train["u2_in"].float().to(device)
        T_val_raw = dict_val["u1_in"].float().to(device)
        kappa_val_raw = dict_val["u2_in"].float().to(device)
        n_query_phys = n_query_phys or n_query

        def _phys_term(z_b, T_raw, kap_raw, snap, create_graph, return_parts=False):
            """Physics loss on one batch: fresh collocation points, source rebuilt at
            the nearest neighbour. create_graph=False -> measurement only."""
            pidx = torch.randint(0, P, (n_query_phys,), device=device)
            y_pde = coords_rte[pidx]
            c_idx = cell_idx_pde[pidx]
            T_at, kap_at = T_raw[snap][:, c_idx], kap_raw[snap][:, c_idx]
            u1_eval = kap_at * IB_COEF * T_at ** 4
            u2_eval = -kap_at
            residual, I_pde = operator.pde_residual(
                [z_b.reshape(z_b.shape[0], -1), None], y_pde, u1_eval, u2_eval,
                create_graph=create_graph)
            return rel_phys_loss(residual, I_pde, u1_eval, u2_eval,
                                 return_parts=return_parts)

        def _phys_measure(z_b, T_raw, kap_raw, snap):
            """(relative, absolute) physics loss as floats -- the tensors die with the
            call, so no graph survives until the next measurement."""
            rel, mse, _ = _phys_term(z_b, T_raw, kap_raw, snap, create_graph=False,
                                     return_parts=True)
            return rel.item(), mse.item()

    if phys_active and loss_balancing == "relobralo":
        from deeponet import ReLoBRaLo
        active_names = [n for n, w in (("data", w_data), ("phys", w_phys)) if w > 0]
        balancer = ReLoBRaLo(active_names, alpha=relobralo_alpha,
                             temperature=relobralo_temp, rho_mean=relobralo_rho,
                             device=device)
        print(f"[operator] ReLoBRaLo active on {active_names} "
              f"(alpha={relobralo_alpha}, tau={relobralo_temp}, E[rho]={relobralo_rho}).")
        lam_current = balancer.lambdas.clone()

    fields_train = encoder._standardize(encoder._stack(dict_train["u1_in"], dict_train["u2_in"]))
    fields_val = encoder._standardize(encoder._stack(dict_val["u1_in"], dict_val["u2_in"]))

    aux = w_recon > 0 and not freeze_encoder
    if w_recon > 0 and freeze_encoder:
        print("[operator] w_recon ignored: the encoder is frozen (no gradient reaches the recon).")

    z_train = z_val = None
    if freeze_encoder:
        operator.ae.requires_grad_(False)
        z_train = _encode_all_tokens(operator, fields_train, encode_batch, device)
        z_val = _encode_all_tokens(operator, fields_val, encode_batch, device)
    else:
        fields_train = fields_train.to(device)
        fields_val = fields_val.to(device)

    dec_params = [p for p in operator.dec_I.parameters() if p.requires_grad]
    if freeze_encoder:
        param_groups = [{"params": dec_params, "lr": lr}]
    else:
        enc_params = [p for p in operator.ae.parameters() if p.requires_grad]
        enc_lr = encoder_lr if encoder_lr is not None else lr
        param_groups = [{"params": dec_params, "lr": lr},
                        {"params": enc_params, "lr": enc_lr}]
        print(f"[operator] Encoder unfrozen: lr dec_I={lr:g}, lr encoder={enc_lr:g}")
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9977)

    N = fields_train.shape[0]
    i_on_dev = False
    if str(device).startswith("cuda"):
        try:
            I_train = I_train.to(device)
            i_on_dev = True
        except RuntimeError:
            torch.cuda.empty_cache()
            print("[operator] I_train kept on CPU (not enough VRAM) -- gathered per step.")
    if freeze_encoder:
        z_train, z_val = z_train.to(device), z_val.to(device)
    gen = torch.Generator().manual_seed(0)
    val_coord_idx = torch.randint(0, P, (min(2 * n_query, P),), generator=gen)
    coords_val = coords_rte[val_coord_idx.to(device)]
    I_val_sub = I_val[:, val_coord_idx].to(device)

    print(f"\n[operator] Training (encoder {'FROZEN' if freeze_encoder else 'FINE-TUNED'}"
          f"{', +recon aux' if aux else ''}{', +phys' if phys_active else ''}, "
          f"M={n_query} coords/pas{f'+{n_query_phys} phys' if phys_active else ''}, "
          f"I_scale={float(operator.I_scale):.3e}, val/{val_every} ep., TF32) on {device} -- {N} snapshots ...")
    best_val, best_state, val_mse = float("inf"), None, float("nan")

    from utils import resume_path, save_atomic
    rpath = resume_path(history_name, "op") if (resume and history_name) else None
    start_epoch = 0
    if rpath and os.path.exists(rpath):
        st = torch.load(rpath, map_location=device, weights_only=False)
        operator.load_state_dict(st["operator"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        best_val, best_state, start_epoch = st["best_val"], st["best_state"], st["epoch"]
        hist.series = st["hist"]
        torch.set_rng_state(st["rng"].cpu())
        print(f"[operator] RESUMING at epoch {start_epoch}/{epochs} "
              f"(meilleure val = {best_val:.6f})")

    for epoch in range(start_epoch, epochs):
        operator.train()
        if freeze_encoder:
            operator.ae.eval()
        perm = torch.randperm(N, device=device)
        ep_loss = ep_loss_data = ep_loss_phys = ep_loss_phys_abs = 0.0
        n_batches = 0
        for b0 in range(0, N, batch_size):
            snap = perm[b0:b0 + batch_size]
            coord_idx = torch.randint(0, P, (n_query,), device=device)
            coords = coords_rte[coord_idx]
            if i_on_dev:
                target = I_train[snap][:, coord_idx]
            else:
                target = I_train[snap.cpu()][:, coord_idx.cpu()].to(device)

            if freeze_encoder:
                z = z_train[snap]
            else:
                z = operator.encode_tokens(fields_train[snap])

            optimizer.zero_grad()
            pred = operator.predict_I(z, coords)
            loss_data = data_loss_fn(pred, target)

            loss_phys = torch.zeros((), device=device)
            if phys_active:
                loss_phys, phys_abs, _ = _phys_term(z, T_train_raw, kappa_train_raw, snap,
                                                    create_graph=True, return_parts=True)
                ep_loss_phys_abs += phys_abs.item()

            if balancer is not None:
                active = [(n, l) for n, l, w in (("data", loss_data, w_data),
                                                 ("phys", loss_phys, w_phys)) if w > 0]
                loss = sum(lam_current[k] * l for k, (_, l) in enumerate(active))
            else:
                loss = w_data * loss_data + w_phys * loss_phys

            if aux:
                c2 = torch.randint(0, n_cells, (n_query,), device=device)
                rec = operator.reconstruct(z, coords_2d[c2])
                tgt2 = fields_train[snap].view(len(snap), 2, -1)[:, :, c2].permute(0, 2, 1)
                loss = loss + w_recon * F.mse_loss(rec, tgt2)

            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            ep_loss_data += loss_data.item()
            ep_loss_phys += loss_phys.item()
            n_batches += 1
        scheduler.step()
        train_loss = ep_loss / max(1, n_batches)
        train_loss_data = ep_loss_data / max(1, n_batches)
        train_loss_phys = ep_loss_phys / max(1, n_batches)
        train_loss_phys_abs = ep_loss_phys_abs / max(1, n_batches)

        if balancer is not None:
            epoch_losses = {"data": train_loss_data, "phys": train_loss_phys}
            lam_current = balancer.update([epoch_losses[n] for n in balancer.names])

        do_val = (epoch + 1) % val_every == 0 or epoch == 0 or epoch == epochs - 1
        phys_train_mon = phys_val_mon = None
        phys_abs_train_mon = phys_abs_val_mon = None
        if do_val:
            operator.eval()
            with torch.no_grad():
                vloss = 0.0
                n_val = z_val.shape[0] if freeze_encoder else fields_val.shape[0]
                for b0 in range(0, n_val, batch_size):
                    sl = slice(b0, b0 + batch_size)
                    zv = z_val[sl] if freeze_encoder else operator.encode_tokens(fields_val[sl])
                    pv = operator.predict_I(zv, coords_val)
                    vloss += (data_loss_fn(pv, I_val_sub[sl]).item() * pv.shape[0] if rel_l2
                              else F.mse_loss(pv, I_val_sub[sl], reduction="sum").item())
                val_mse = vloss / n_val if rel_l2 else vloss / I_val_sub.numel() / scale_sq

            if monitor_phys:
                snap_m = perm[:batch_size]
                with torch.no_grad():
                    z_m = (z_train[snap_m] if freeze_encoder
                           else operator.encode_tokens(fields_train[snap_m]))
                phys_train_mon, phys_abs_train_mon = _phys_measure(
                    z_m, T_train_raw, kappa_train_raw, snap_m)
                snap_v = torch.arange(min(batch_size, n_val), device=device)
                with torch.no_grad():
                    z_v = (z_val[snap_v] if freeze_encoder
                           else operator.encode_tokens(fields_val[snap_v]))
                phys_val_mon, phys_abs_val_mon = _phys_measure(
                    z_v, T_val_raw, kappa_val_raw, snap_v)

            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in operator.state_dict().items()}
                if save_path is not None:
                    save_operator(operator, encoder, save_path,
                                  meta={"best_val": best_val, "epoch": epoch + 1}, verbose=False)

        hist.log(epoch + 1, train_total=train_loss, train_data=train_loss_data,
                 train_phys=(train_loss_phys if phys_active else phys_train_mon),
                 train_phys_abs=(train_loss_phys_abs if phys_active else phys_abs_train_mon),
                 val_data=(val_mse if do_val else None), val_phys=phys_val_mon,
                 val_phys_abs=phys_abs_val_mon)

        if rpath and do_val:
            save_atomic({"epoch": epoch + 1, "operator": operator.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "best_val": best_val, "best_state": best_state,
                         "hist": hist.series, "rng": torch.get_rng_state()}, rpath)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            vtxt = f"{val_mse:.6f}" if do_val else "   --   "
            print(f"  [operator] Epoch {epoch+1:4d}/{epochs} | Train: {train_loss:.6f} "
                  f"| Val I-MSE(rel): {vtxt} | Best: {best_val:.6f}")
            if phys_active or phys_train_mon is not None:
                p_tr = train_loss_phys if phys_active else phys_train_mon
                extra = f"           | data={train_loss_data:.6f} phys={p_tr:.6f}"
                if not phys_active:
                    extra += " (suivi seul)"
                if phys_val_mon is not None:
                    extra += f" | val phys={phys_val_mon:.6f}"
                if balancer is not None:
                    extra += " | λ " + " ".join(
                        f"{n}={lam_current[k]:.3f}" for k, n in enumerate(balancer.names))
                print(extra)

    if rpath and os.path.exists(rpath):
        os.remove(rpath)
    if best_state is not None:
        operator.load_state_dict(best_state)
        print(f"  [operator] Best model reloaded (Val relative I-MSE = {best_val:.6f})")
    if history_name is not None:
        hist.meta["best_val"] = best_val
        hist.save()
        try:
            from utils import plot_learning_curves
            plot_learning_curves(history_name, title=f"{history_name} — learning curves")
        except Exception as e:
            print(f"  [!] learning-curve plot skipped: {e}")
    operator.eval()
    return operator


def save_operator(operator, encoder, path, meta=None, verbose=True):
    """Save weights, input stats, I_scale and architecture of the operator."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mean, std = encoder.stats
    arch = {"patch": encoder.patch, "dim": encoder.dim, "n_latent": encoder.n_latent,
            "depth": encoder.depth, "dec_depth": encoder.dec_depth, "heads": encoder.heads,
            "mlp_ratio": encoder.mlp_ratio, "n_fourier": encoder.n_fourier,
            "fourier_sigma": encoder.fourier_sigma}
    state = {
        "kind": "tbd_operator", "grid_shape": encoder.grid_shape, "arch": arch,
        "ae": operator.ae.state_dict(), "dec_I": operator.dec_I.state_dict(),
        "I_scale": float(operator.I_scale), "affine": bool(operator.affine),
        "stats": (mean.cpu(), std.cpu()),
        "xy_lo": operator.xy_lo.cpu(), "xy_hi": operator.xy_hi.cpu(),
        "meta": meta or {},
    }
    from utils import save_atomic
    save_atomic(state, path)
    if verbose:
        print(f"Transformer-Based DeepONet (T,kappa)->I operator saved to: {path}")


def load_operator(path, device="cpu"):
    """Load (operator, encoder_wrapper), the Evaluator's model and encoder."""
    state = torch.load(path, map_location=device)
    arch = state["arch"]
    gs = state["grid_shape"]
    ae = FunctionAutoencoder(grid_shape=gs, in_channels=2, **arch).to(device)
    ae.load_state_dict(state["ae"])
    operator = TBDOperator(ae, I_scale=state["I_scale"], dec_depth=arch["dec_depth"],
                           heads=arch["heads"], mlp_ratio=arch["mlp_ratio"],
                           n_fourier=arch["n_fourier"], fourier_sigma=arch["fourier_sigma"],
                           affine=bool(state.get("affine", False))).to(device)
    operator.dec_I.load_state_dict(state["dec_I"])
    operator.xy_lo.copy_(state["xy_lo"].to(device))
    operator.xy_hi.copy_(state["xy_hi"].to(device))
    operator.eval()

    encoder = TBDEncoder(grid_shape=gs, patch=arch["patch"], dim=arch["dim"],
                              n_latent=arch["n_latent"], depth=arch["depth"],
                              dec_depth=arch["dec_depth"], heads=arch["heads"],
                              mlp_ratio=arch["mlp_ratio"], n_fourier=arch["n_fourier"],
                              fourier_sigma=arch["fourier_sigma"], device=device)
    encoder.model = ae
    mean, std = state["stats"]
    encoder.stats = (mean.to(device), std.to(device))
    encoder.fitted = True
    return operator, encoder


