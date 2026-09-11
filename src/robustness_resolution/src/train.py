import argparse
import importlib.util
import os
import sys

import torch

_SRC = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(os.path.dirname(_SRC))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _SRC]
sys.path.insert(0, _PARENT)
_spec = importlib.util.spec_from_file_location("rr_utils", os.path.join(_SRC, "utils.py"))
U = importlib.util.module_from_spec(_spec)
sys.modules["rr_utils"] = U
_spec.loader.exec_module(U)


def train_tbd(args, device):
    U.build_mixed()
    U.fit_tbd_encoder_on_mixed(device, epochs=args.encoder_epochs)
    art = U.load_mixed()
    groups = art["groups"]
    is_train = U.split_mask(art["meta"]["N"], "train", art["meta"]["split_seed"])

    encoder = U.load_encoder_for("tbd", device)
    finetune = args.finetune_encoder
    print(f"[Transformer-Based DeepONet] encodeur patch={encoder.patch} "
          f"(latent {encoder.latent_dim}), {'FINE-TUNED' if finetune else 'frozen'}")
    latents = None if finetune else U.compute_tbd_latents(encoder, groups, device)
    I_scale = U.median_I_scale(groups, is_train)
    operator = U.build_tbd_operator(encoder, I_scale, art["xy_lo"], art["xy_hi"], device,
                                    freeze_encoder=not finetune)

    T_val, k_val, I_val, y150, _ = U.load_clean_split("val")
    val_pack = U.make_tbd_val_pack(encoder, T_val, k_val, I_val, y150, device, args.n_query)

    save_fn = lambda op, meta: U.save_tbd(op, encoder, args.name, meta)
    operator, best = U.fit_tbd(operator, latents, groups, is_train, val_pack, I_scale, device,
                               epochs=args.epochs, batch_size=args.batch_size,
                               n_query=args.n_query, lr=args.lr, val_every=args.val_every,
                               grad_clip=args.grad_clip, save_fn=save_fn,
                               data_loss=args.data_loss, history_name=args.name,
                               phys_val=(T_val, k_val, y150),
                               monitor_phys=args.monitor_phys,
                               n_query_phys=args.n_query_phys,
                               finetune_encoder=finetune, encoder_lr=args.encoder_lr,
                               stats=encoder.stats)
    U.save_tbd(operator, encoder, args.name, meta={"best_val_I_l2": best, "epochs": args.epochs})
    print(f"[Transformer-Based DeepONet] done -- best val I-L2 = {best:.4f} "
          f"-> {U.tbd_ckpt(args.name)}")


def train_modular(args, device):
    """PCA/AE DeepONet at a fixed working resolution (`--input_res` 150 or 50)."""
    from deeponet import (MultiBranchDeepONet, Trainer, FullGridDataset, INPUT_KEYS)
    from utils import QUERY_DIM
    R = args.input_res
    U.build_mixed()
    dataset = U.build_interp150() if R == U.GRID else U.build_fixed_res(R)
    U.fit_encoder_at_res(args.model, R, dataset, device, ae_epochs=args.encoder_epochs)

    splits = U.load_and_split(dataset)
    dict_pde_tr, dict_pde_va = splits["pde"]
    dict_bc_tr, dict_bc_va = splits["bc"]
    data_tr, data_va = splits["data"]
    y_full = splits["y_full"]
    print(f"[{args.model}@{R}²] split 70/15/15 : {data_tr['u1_in'].shape[0]} train / "
          f"{data_va['u1_in'].shape[0]} val / {splits['data_test']['u1_in'].shape[0]} test")

    encoder = U.load_encoder_for(args.model, device, res=R)
    finetune = args.finetune_encoder and args.model == "ae"
    print(f"[{args.model}@{R}^2] encoder {'FINE-TUNED' if finetune else 'frozen'}, "
          f"latents {encoder.latent_dims}")
    if finetune:
        # Raw fields: the AE must run inside the graph to be fine-tuned.
        train_ds = FullGridDataset(data_tr["u1_in"], data_tr["u2_in"], data_tr["I_target"], y_full)
        val_ds = FullGridDataset(data_va["u1_in"], data_va["u2_in"], data_va["I_target"], y_full)
    else:
        for d in (dict_pde_tr, dict_pde_va, dict_bc_tr, dict_bc_va):
            enc = encoder.transform_dicts(d, INPUT_KEYS)
            d["u1_in"], d["u2_in"] = enc["u1_in"], enc["u2_in"]
        enc_tr = encoder.transform_dicts({"u1_in": data_tr["u1_in"], "u2_in": data_tr["u2_in"]}, INPUT_KEYS)
        enc_va = encoder.transform_dicts({"u1_in": data_va["u1_in"], "u2_in": data_va["u2_in"]}, INPUT_KEYS)
        train_ds = FullGridDataset(enc_tr["u1_in"], enc_tr["u2_in"], data_tr["I_target"], y_full)
        val_ds = FullGridDataset(enc_va["u1_in"], enc_va["u2_in"], data_va["I_target"], y_full)

    I_nz = data_tr["I_target"][data_tr["I_target"] > 1e-3]
    I_scale = float(I_nz.median()) if I_nz.numel() else 1.0
    print(f"[{args.model}] I_scale={I_scale:.4e} | {len(train_ds)} train / {len(val_ds)} val")

    model = MultiBranchDeepONet(
        branch_input_dims=[encoder.latent_dims["u1_in"], encoder.latent_dims["u2_in"]],
        trunk_input_dim=QUERY_DIM, p_basis=args.p_basis, trunk_hidden=[512, 512, 512, 512],
        I_scale_val=I_scale, affine=args.affine).to(device)
    print("[head] I = (<beta, gamma(y)> + b) * I_scale -- AFFINE (Thm 3.4)" if args.affine
          else "[head] I = softplus(<beta, gamma(y)> + b) * I_scale")

    os.makedirs(U.CKPT_DIR, exist_ok=True)
    trainer = Trainer(model, device, save_dir=U.CKPT_DIR)
    trainer.train(dict_pde_tr, dict_bc_tr, train_ds, dict_pde_va, dict_bc_va, val_ds,
                  epochs=args.epochs, batch_size_pde=2024, batch_size_data=args.batch_size_data,
                  w_phys=0.0, w_bc=0.0, w_data=1.0, save_name=args.name,
                  n_colloc=0, colloc_sampler=None, loss_balancing="static",
                  encoder=encoder, finetune_encoder=finetune, unfreeze_encoder_epoch=0,
                  encoder_lr=args.encoder_lr, base_lr=args.lr, grad_clip=args.grad_clip,
                  data_loss=args.data_loss, monitor_phys=args.monitor_phys,
                  monitor_every=args.val_every)
    print(f"[{args.model}] done -> {U.modular_ckpt(args.name)}")


def parse_args():
    p = argparse.ArgumentParser(description="Train one model (mixed-resolution robustness)")
    p.add_argument("--model", required=True, choices=["tbd", "pca", "ae"])
    p.add_argument("--name", default=None,
                   help="checkpoint name (default: res-<MODEL>[-<res>])")
    p.add_argument("--input_res", type=int, default=150,
                   help="Working resolution of the FIXED-dimension models (pca/ae): "
                        "150 (coarse data upsampled) or 50 (everything brought down to "
                        "50^2, high resolution lost). No effect on the "
                        "Transformer-Based DeepONet (native).")
    p.add_argument("--epochs", type=int, default=None,
                   help="default: Transformer-Based DeepONet 1000, PCA/AE 2000")
    p.add_argument("--encoder_epochs", type=int, default=1000,
                   help="Stage one: reconstruction pretraining of the encoder ON THE "
                        "MIXED DATA (no effect on pca, a direct fit).")
    p.add_argument("--finetune_encoder", action=argparse.BooleanOptionalAction, default=True,
                   help="ae and tbd: fine-tune the encoder jointly with the "
                        "reconstructor at --encoder_lr, as the appendix states and as "
                        "the fixed-resolution campaign does. A PCA has no weights to "
                        "train, so the flag is a no-op there.")
    p.add_argument("--encoder_lr", type=float, default=1e-4,
                   help="LR of the encoder weights while fine-tuned.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--p_basis", type=int, default=128)
    p.add_argument("--affine", action="store_true",
                   help="PCA/AE only: drop the output softplus, so the reconstructor is "
                        "affine in beta and the bound of Theorem 3.4 applies to the "
                        "letter. This is what the paper reports.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_query", type=int, default=8192)
    p.add_argument("--val_every", type=int, default=25)
    p.add_argument("--batch_size_data", type=int, default=256)
    p.add_argument("--data_loss", default="rel_l2", choices=["rel_l2", "mse"],
                   help="Data loss: per-snapshot rel-L2 (default) or MSE.")
    p.add_argument("--monitor_phys", action=argparse.BooleanOptionalAction, default=True,
                   help="Measure -- without optimising -- the RTE residual at the "
                        "validation cadence, on both families, so the learning curves "
                        "carry the same panels. --no-monitor_phys to switch it off.")
    p.add_argument("--n_query_phys", type=int, default=2048,
                   help="Collocation points per physics measurement (Transformer-Based "
                        "DeepONet only; 8192 saturates the GPU).")
    a = p.parse_args()
    if a.affine and a.model == "tbd":
        p.error("--affine only applies to the fixed-dimension models (pca/ae)")
    if a.name is None:
        base = {"tbd": "res-TBD", "pca": "res-PCA", "ae": "res-AE"}[a.model]
        a.name = base if a.model == "tbd" else f"{base}-{a.input_res}"
    if a.epochs is None:
        a.epochs = 1000 if a.model == "tbd" else 2000
    return a


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAIN {args.model} ({args.name}, {args.epochs} ep.) on {device.type.upper()} ===")
    if args.model == "tbd":
        train_tbd(args, device)
    else:
        train_modular(args, device)
