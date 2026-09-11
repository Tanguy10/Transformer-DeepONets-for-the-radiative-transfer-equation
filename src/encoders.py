import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import CHECKPOINTS_DIR


class BaseEncoder:
    """Common interface of every branch-net input encoder."""

    kind = "base"

    def __init__(self):
        self.latent_dims = {}
        self.fitted = False

    def fit_transform_dicts(self, dict_train, input_keys, dict_val=None):
        raise NotImplementedError

    def transform_dicts(self, dict_val, input_keys):
        raise NotImplementedError

    def reconstruct_field(self, key, X):
        """Rebuild a raw field (N, N_cells): encode, then decode."""
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path, device="cpu"):
        raise NotImplementedError

class PCAEncoder(BaseEncoder):
    """PCA applied independently to each input field.

    use_scaler: StandardScaler + PCA; otherwise raw, centred PCA (POD)."""

    kind = "pca"

    def __init__(self, n_components=500, use_scaler=True):
        super().__init__()
        self.n_components = int(n_components)
        self.use_scaler = use_scaler
        self.scalers = {}
        self.pcas = {}

    def fit_transform_dicts(self, dict_train, input_keys=("u1_in", "u2_in"), dict_val=None):
        """Fit the PCA on the training data and transform it (`dict_val` is ignored)."""
        dict_transformed = dict_train.copy()

        for key in input_keys:
            X = dict_train[key].cpu().numpy()
            X_flat = X.reshape(X.shape[0], -1)

            scaler = StandardScaler() if getattr(self, "use_scaler", True) else None
            pca = PCA(n_components=self.n_components)

            X_scaled = scaler.fit_transform(X_flat) if scaler is not None else X_flat
            X_latent = pca.fit_transform(X_scaled)

            self.scalers[key] = scaler
            self.pcas[key] = pca
            self.latent_dims[key] = pca.n_components_

            dict_transformed[key] = torch.tensor(X_latent, dtype=torch.float32)
            print(f"[{key}] PCA fit: input dim={X_flat.shape[1]} -> latent d={pca.n_components_}")

        self.fitted = True
        return dict_transformed

    def transform_dicts(self, dict_val, input_keys=("u1_in", "u2_in")):
        """Transform data WITHOUT refitting the PCA (avoids leakage)."""
        if not self.fitted:
            raise ValueError("PCAEncoder must be fitted on the training data first.")

        dict_transformed = dict_val.copy()
        for key in input_keys:
            X = dict_val[key].cpu().numpy()
            X_flat = X.reshape(X.shape[0], -1)
            scaler = self.scalers[key]
            X_scaled = scaler.transform(X_flat) if scaler is not None else X_flat
            X_latent = self.pcas[key].transform(X_scaled)
            dict_transformed[key] = torch.tensor(X_latent, dtype=torch.float32)
        return dict_transformed

    def reconstruct_field(self, key, X):
        """Rebuild a raw field: scaler -> PCA -> inverse PCA -> inverse scaler."""
        if not self.fitted:
            raise ValueError("PCAEncoder must be fitted first.")
        X_np = torch.as_tensor(X).cpu().numpy()
        X_flat = X_np.reshape(X_np.shape[0], -1)
        scaler = self.scalers[key]
        X_scaled = scaler.transform(X_flat) if scaler is not None else X_flat
        X_latent = self.pcas[key].transform(X_scaled)
        X_rec_scaled = self.pcas[key].inverse_transform(X_latent)
        X_rec = scaler.inverse_transform(X_rec_scaled) if scaler is not None else X_rec_scaled
        return torch.tensor(X_rec, dtype=torch.float32)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"PCA encoder saved to: {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        with open(path, "rb") as f:
            return pickle.load(f)

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=500, base=32, grid_shape=(150, 150)):
        super().__init__()
        self.base = base
        H, W = grid_shape
        if H % 10 or W % 10:
            raise ValueError(f"grid_shape {grid_shape} must be divisible by 10 (pools of 2 then 5)")
        self.grid_shape = (H, W)
        self.fh, self.fw = H // 10, W // 10
        def enc_block(ci, co, pool):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                nn.MaxPool2d(pool))
        self.encoder_conv = nn.Sequential(enc_block(1, base, 2), enc_block(base, base*2, 5))
        self.enc_fc = nn.Linear(base*2*self.fh*self.fw, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, base*2*self.fh*self.fw)
        def dec_block(ci, co, scale):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())
        self.decoder_conv = nn.Sequential(
            dec_block(base*2, base, 5), dec_block(base, base, 2),
            nn.Conv2d(base, 1, 3, padding=1))

    def encode(self, x):
        """Encoder half only: field (N,1,H,W) -> latent vector (N, latent_dim)."""
        h = self.encoder_conv(x)
        return self.enc_fc(h.flatten(1))

    def forward(self, x):
        z = self.encode(x)
        out = self.dec_fc(z).view(-1, self.base*2, self.fh, self.fw)
        return self.decoder_conv(out), z


class AutoencoderEncoder(BaseEncoder):
    """One convolutional autoencoder per input field; the branch nets read its latent."""

    kind = "autoencoder"

    def __init__(self, latent_dim=128, grid_shape=(150, 150),
                 epochs=200, batch_size=64, lr=1e-3, device=None, encode_batch=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_shape = tuple(grid_shape)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encode_batch = encode_batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        self.stats = {}

    def _make_model(self):
        """Instantiate an untrained model. Overridden to swap architecture."""
        return ConvAutoencoder(latent_dim=self.latent_dim, grid_shape=self.grid_shape)

    def _loss(self, pred, target):
        """Reconstruction loss. Overridden to add a term (e.g. GDL)."""
        return nn.functional.mse_loss(pred, target)

    def _extra_state(self):
        """Extra fields to serialise (constructor args specific to the variant)."""
        return {}

    @classmethod
    def _model_from_state(cls, state):
        """Rebuild the right model on load (overridden by the variants)."""
        return ConvAutoencoder(latent_dim=state["latent_dim"],
                               grid_shape=tuple(state.get("grid_shape", (150, 150))))

    def fit_transform_dicts(self, dict_train, input_keys=("u1_in", "u2_in"), dict_val=None):
        dict_transformed = dict_train.copy()
        for key in input_keys:
            X = dict_train[key].float()
            data = X.view(-1, 1, *self.grid_shape)

            mean_val = data.mean()
            std_val = data.std()
            data_scaled = (data - mean_val) / std_val

            val_loader = None
            if dict_val is not None and key in dict_val:
                data_val_scaled = (dict_val[key].float().view(-1, 1, *self.grid_shape) - mean_val) / std_val
                val_loader = DataLoader(TensorDataset(data_val_scaled), batch_size=self.batch_size, shuffle=False)

            model = self._make_model().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            loader = DataLoader(TensorDataset(data_scaled), batch_size=self.batch_size, shuffle=True)

            print(f"\n[{key}] Training autoencoder (latent d={self.latent_dim}) on {self.device} ...")
            best_val, best_state = float("inf"), None
            for epoch in range(self.epochs):
                model.train()
                epoch_loss = 0.0
                for (batch,) in loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    reconstructed, _ = model(batch)
                    loss = self._loss(reconstructed, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                train_mse = epoch_loss / len(loader)

                val_mse = None
                if val_loader is not None:
                    model.eval()
                    with torch.no_grad():
                        v_loss = sum(self._loss(model(vb.to(self.device))[0], vb.to(self.device)).item()
                                     for (vb,) in val_loader)
                    val_mse = v_loss / len(val_loader)
                    if val_mse < best_val:
                        best_val = val_mse
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    if val_mse is not None:
                        print(f"  [{key}] Epoch {epoch+1:3d}/{self.epochs} | Train MSE: {train_mse:.6f} | "
                              f"Val MSE: {val_mse:.6f} | Best Val: {best_val:.6f}")
                    else:
                        print(f"  [{key}] Epoch {epoch+1:3d}/{self.epochs} | Loss MSE: {train_mse:.6f}")

            if best_state is not None:
                model.load_state_dict(best_state)
                print(f"  [{key}] Best model reloaded (Val MSE = {best_val:.6f})")
            model.eval()

            self.models[key] = model
            self.stats[key] = (mean_val.detach().cpu(), std_val.detach().cpu())
            self.latent_dims[key] = self.latent_dim

            dict_transformed[key] = self._encode(key, X)

        self.fitted = True
        return dict_transformed

    def transform_dicts(self, dict_val, input_keys=("u1_in", "u2_in")):
        if not self.fitted:
            raise ValueError("AutoencoderEncoder must be fitted (train_encoder) first.")
        dict_transformed = dict_val.copy()
        for key in input_keys:
            dict_transformed[key] = self._encode(key, dict_val[key].float())
        return dict_transformed

    def _encode(self, key, X):
        """Encode raw fields (N, N_cells) into latent vectors (N, latent_dim)."""
        model = self.models[key]
        mean_val, std_val = self.stats[key]
        model.eval()

        data = (X.view(-1, 1, *self.grid_shape) - mean_val) / std_val
        outputs = []
        with torch.no_grad():
            for i in range(0, data.shape[0], self.encode_batch):
                chunk = data[i:i + self.encode_batch].to(self.device)
                z = model.encode(chunk)
                outputs.append(z.cpu())
        return torch.cat(outputs, dim=0).float()

    def encode_torch(self, key, X):
        """Differentiable encoding (N, N_cells) -> (N, latent_dim), on X's device."""
        model = self.models[key]
        mean_val, std_val = self.stats[key]
        data = (X.float().view(-1, 1, *self.grid_shape) - mean_val.to(X.device)) / std_val.to(X.device)
        return model.encode(data)

    def parameters(self):
        """Trainable parameters of every AE sub-model (one per field)."""
        for m in self.models.values():
            yield from m.parameters()

    def requires_grad_(self, flag):
        """Freeze or unfreeze every AE weight (train/eval mode untouched)."""
        for m in self.models.values():
            for p in m.parameters():
                p.requires_grad_(flag)

    def train_mode(self):
        for m in self.models.values():
            m.train()

    def eval_mode(self):
        for m in self.models.values():
            m.eval()

    def init_untrained(self, dict_train, input_keys=("u1_in", "u2_in")):
        """Untrained AEs and standardisation stats, for from-scratch joint training."""
        for key in input_keys:
            data = dict_train[key].float().view(-1, 1, *self.grid_shape)
            self.stats[key] = (data.mean().detach().cpu(), data.std().detach().cpu())
            self.models[key] = self._make_model().to(self.device)
            self.latent_dims[key] = self.latent_dim
        self.fitted = True
        return self

    def reconstruct_field(self, key, X):
        """Rebuild a raw field: standardise -> AE (encode+decode) -> de-standardise."""
        model = self.models[key]
        mean_val, std_val = self.stats[key]
        model.eval()

        data = (torch.as_tensor(X).float().view(-1, 1, *self.grid_shape) - mean_val) / std_val
        outputs = []
        with torch.no_grad():
            for i in range(0, data.shape[0], self.encode_batch):
                chunk = data[i:i + self.encode_batch].to(self.device)
                rec, _ = model(chunk)
                outputs.append(rec.cpu())
        rec = torch.cat(outputs, dim=0).view(data.shape[0], -1)
        rec = rec * std_val + mean_val
        return rec.float()

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "kind": self.kind,
            "latent_dim": self.latent_dim,
            "grid_shape": self.grid_shape,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "latent_dims": self.latent_dims,
            "models": {k: m.state_dict() for k, m in self.models.items()},
            "stats": {k: (float(mu), float(sd)) for k, (mu, sd) in self.stats.items()},
            "extra": self._extra_state(),
        }
        torch.save(state, path)
        print(f"Autoencoder encoder saved to: {path}")

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device)
        enc = cls(
            latent_dim=state["latent_dim"],
            grid_shape=state["grid_shape"],
            epochs=state.get("epochs", 200),
            batch_size=state.get("batch_size", 64),
            lr=state.get("lr", 1e-3),
            device=device,
            **state.get("extra", {}),
        )
        enc.latent_dims = state["latent_dims"]
        for k, sd in state["models"].items():
            m = cls._model_from_state(state).to(device)
            m.load_state_dict(sd)
            m.eval()
            enc.models[k] = m
        enc.stats = {k: (torch.tensor(mu), torch.tensor(sd)) for k, (mu, sd) in state["stats"].items()}
        enc.fitted = True
        return enc


_ENCODER_REGISTRY = {
    "pca": PCAEncoder,
    "autoencoder": AutoencoderEncoder,
}

_ENCODER_EXT = {
    "pca": "pkl",
    "autoencoder": "pt",
}


def _resolve_encoder_cls(kind):
    """Return the encoder class registered under `kind`."""
    if kind in _ENCODER_REGISTRY:
        return _ENCODER_REGISTRY[kind]
    raise ValueError(f"Unknown encoder: '{kind}'. Choose from {list(_ENCODER_REGISTRY)}")


def make_encoder(kind, **cfg):
    """Build an encoder of the requested kind, keeping only the relevant kwargs."""
    if kind == "pca":
        allowed = {"n_components", "use_scaler"}
    elif kind == "autoencoder":
        allowed = {"latent_dim", "grid_shape", "epochs", "batch_size", "lr", "device", "encode_batch"}
    else:
        _resolve_encoder_cls(kind)

    cls = _resolve_encoder_cls(kind)
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    return cls(**filtered)


HEAD_SUFFIXES = ("-softplus", "-affine")


def encoder_run(name):
    """The configuration whose encoder `name` reads: the name without its head suffix.

    Fine-tuned weights are saved under the full run name, as `<run>_ft`."""
    for suffix in HEAD_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def encoder_path(name, kind, save_dir=CHECKPOINTS_DIR):
    """Standard encoder path: checkpoints/encoder_<kind>_<name>.<ext>."""
    ext = _ENCODER_EXT[kind]
    return os.path.join(save_dir, f"encoder_{kind}_{encoder_run(name)}.{ext}")


def load_encoder(kind, path, device="cpu"):
    """Load an encoder of the requested kind from disk."""
    return _resolve_encoder_cls(kind).load(path, device=device)

