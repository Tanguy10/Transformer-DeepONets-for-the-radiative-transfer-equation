#!/usr/bin/env python
"""Single entry point to train the three operator families of this repository.

    python src/train.py --model pca --name my-run --p_basis 512
    python src/train.py --model ae  --name my-run --p_basis 128
    python src/train.py --model tbd --name my-run --n_latent 4 --dim 128

Every model is trained in two stages, run back to back by default (`--stage all`):

  1. `encoder`   fit the input encoder on the (T, kappa) fields of the TRAIN split
                 only -- PCA basis, convolutional autoencoder, or Transformer-Based DeepONet.
  2. `operator`  train the (T, kappa) -> I operator on top of it.

This script does not implement anything: it translates its own flags into the
underlying CLI of `deeponet.py` (PCA / AE) or `deeponet_joint.py` (Transformer-Based
DeepONet) and calls the very same functions those scripts call. Any option not exposed
here keeps the default of the underlying parser, so numbers are reproducible either way. The
defaults below are exactly the ones of the reference campaign (see README).
"""
import argparse
import copy
import os
import sys

import torch

MODELS = ("pca", "ae", "tbd")
ENCODER_KIND = {"pca": "pca", "ae": "autoencoder"}


def encoder_checkpoint(model, name):
    """Path the encoder of `<model>/<name>` is written to."""
    if model == "tbd":
        from deeponet_joint import joint_encoder_path
        return joint_encoder_path(name, "tbd")
    from encoders import encoder_path
    return encoder_path(name, ENCODER_KIND[model])


def operator_checkpoint(model, name):
    """Path the trained operator of `<model>/<name>` is written to."""
    if model == "tbd":
        from deeponet_joint import operator_path
        return operator_path(name)
    from utils import CHECKPOINTS_DIR
    return os.path.join(CHECKPOINTS_DIR, f"{name}.pth")


def _encoder_argv(args):
    """Underlying CLI for stage 1."""
    if args.model == "tbd":
        argv = ["--mode", "train_encoder", "--joint_encoder", "tbd",
                "--name", args.name, "--dataset", args.dataset,
                "--ae_epochs", str(args.encoder_epochs),
                "--ae_batch_size", str(args.encoder_batch_size),
                "--ae_lr", str(args.encoder_pretrain_lr),
                "--tbd_n_latent", str(args.n_latent), "--tbd_dim", str(args.dim),
                "--tbd_patch", str(args.patch), "--tbd_depth", str(args.depth),
                "--tbd_dec_depth", str(args.dec_depth), "--tbd_heads", str(args.heads),
                "--tbd_n_fourier", str(args.n_fourier), "--tbd_sigma", str(args.sigma),
                "--tbd_n_query", str(args.encoder_n_query),
                "--tbd_subsample", args.subsample]
        return argv + (["--resume"] if args.resume else ["--no-resume"])

    argv = ["--mode", "train_encoder", "--encoder", ENCODER_KIND[args.model],
            "--name", args.name, "--dataset", args.dataset,
            "--ae_epochs", str(args.encoder_epochs),
            "--ae_batch_size", str(args.encoder_batch_size),
            "--ae_lr", str(args.encoder_pretrain_lr),
            "--ae_latent_dim", str(args.latent_dim),
            "--pca_components", str(args.pca_components)]
    if args.pca_no_scaler:
        argv.append("--pca_no_scaler")
    return argv


def _operator_argv(args):
    """Underlying CLI for stage 2."""
    if args.model == "tbd":
        argv = ["--mode", "train_operator", "--name", args.name,
                "--dataset", args.dataset, "--pretrained_encoder", args.pretrained_encoder,
                "--op_epochs", str(args.epochs), "--op_lr", str(args.lr),
                "--op_batch", str(args.batch_size), "--op_n_query", str(args.n_query),
                "--op_n_query_phys", str(args.n_query_phys),
                "--op_val_every", str(args.val_every), "--data_loss", args.data_loss,
                "--w_data", str(args.w_data), "--w_phys", str(args.w_phys),
                "--w_bc", str(args.w_bc), "--loss_balancing", args.loss_balancing,
                "--op_data_frac", str(args.data_frac)]
        if args.affine:
            argv.append("--affine")
        argv += ["--freeze_encoder"] if args.freeze_encoder else \
                ["--no-freeze_encoder", "--op_encoder_lr", str(args.encoder_lr)]
        return argv + (["--resume"] if args.resume else ["--no-resume"])

    argv = ["--mode", "train_deeponet", "--encoder", ENCODER_KIND[args.model],
            "--name", args.name, "--dataset", args.dataset,
            "--epochs", str(args.epochs), "--lr", str(args.lr),
            "--batch_size_data", str(args.batch_size), "--p_basis", str(args.p_basis),
            "--data_loss", args.data_loss, "--w_data", str(args.w_data),
            "--w_phys", str(args.w_phys), "--w_bc", str(args.w_bc),
            "--n_colloc", str(args.n_colloc), "--grad_clip", str(args.grad_clip),
            "--batch_size_pde", str(args.batch_size_pde),
            "--loss_balancing", args.loss_balancing]
    if args.affine:
        argv.append("--affine")
    if args.model == "ae" and not args.freeze_encoder:
        argv += ["--finetune_encoder", "--encoder_lr", str(args.encoder_lr)]
    return argv + (["--resume"] if args.resume else ["--no-resume"])


def run_stage(args, stage):
    """Parse the underlying CLI with ITS own parser (so every unexposed option keeps
    its documented default) and call the matching entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "tbd":
        import deeponet_joint as backend
        argv = _encoder_argv(args) if stage == "encoder" else _operator_argv(args)
        sub = backend.build_parser().parse_args(argv)
        entry = backend.run_train_encoder if stage == "encoder" else backend.run_train_operator
    else:
        import deeponet as backend
        argv = _encoder_argv(args) if stage == "encoder" else _operator_argv(args)
        sub = backend.build_parser().parse_args(argv)
        entry = backend.run_train_encoder if stage == "encoder" else backend.run_train_deeponet

    if sub.output_dir is None:
        from utils import RESULTS_DIR
        sub.output_dir = os.path.join(RESULTS_DIR, sub.name)

    script = "deeponet_joint.py" if args.model == "tbd" else "deeponet.py"
    print(f"\n{'='*78}\n[train.py] stage {stage.upper()} -> {script} "
          f"{' '.join(argv)}\n{'='*78}")
    # Both run_train_encoder take args only; the encoders pick their own device.
    return entry(sub) if stage == "encoder" else entry(sub, device)


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a PCA-DeepONet, AE-DeepONet or Transformer-Based DeepONet "
                    "on the 2D RTE data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model", required=True, choices=MODELS,
                   help="pca/ae: DeepONet with a per-field encoder and a Hadamard "
                        "product of branch nets. tbd: FunDiff encoder + coordinate "
                        "cross-attention decoder.")
    p.add_argument("--name", required=True,
                   help="Run name. Drives every path: checkpoints/<name>.pth (or "
                        "operator_tbd_<name>.pt) and results/<name>/.")
    p.add_argument("--stage", default="all", choices=["all", "encoder", "operator"],
                   help="Which half of the pipeline to run.")
    p.add_argument("--dataset", default=None, help="Path to the .pt dataset "
                                                   "(default: data/dataset_train_full_phi.pt)")

    g = p.add_argument_group("operator (stage 2)")
    g.add_argument("--epochs", type=int, default=2000, help="Operator epochs.")
    g.add_argument("--lr", type=float, default=1e-3,
                   help="Initial LR of the operator, decayed by ExponentialLR(0.9977).")
    g.add_argument("--batch_size", type=int, default=None,
                   help="Snapshots per step (default: 256 for pca/ae, 64 for tbd).")
    g.add_argument("--data_loss", default="rel_l2", choices=["rel_l2", "mse"],
                   help="rel_l2 = per-snapshot relative L2 (also the reported metric).")
    g.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=False,
                   help="Keep the pretrained encoder frozen while training the "
                        "operator. Default: fine-tune it at --encoder_lr. Has no "
                        "effect for pca (a PCA basis has no trainable weights).")
    g.add_argument("--encoder_lr", type=float, default=1e-4,
                   help="LR of the encoder weights while fine-tuning.")

    g = p.add_argument_group("DeepONet only (pca, ae)")
    g.add_argument("--p_basis", type=int, default=128,
                   help="Size of the latent basis shared by the branch and trunk nets.")
    g.add_argument("--batch_size_pde", type=int, default=64,
                   help="Snapshots per physics step. The residual is evaluated at every "
                        "collocation point FOR EACH of them, and under --no-freeze_encoder "
                        "each one also goes through the conv AE inside the graph: the 2024 "
                        "of deeponet.py OOMs from 256 up. Ignored when "
                        "--w_phys = 0 (the monitoring caps it at 64 by itself).")
    g.add_argument("--n_colloc", type=int, default=0,
                   help="PDE collocation points resampled per epoch (0 = use the "
                        "stored pool). Only matters when --w_phys > 0 or when the "
                        "physics residual is monitored.")
    g.add_argument("--grad_clip", type=float, default=0.0,
                   help="Gradient-norm clipping (0 = off).")

    g = p.add_argument_group("Transformer-Based DeepONet only")
    g.add_argument("--n_latent", "-N", type=int, default=4,
                   help="Number of latent tokens N. Joint latent size is N*dim.")
    g.add_argument("--dim", "-D", type=int, default=128, help="Token width D.")
    g.add_argument("--patch", type=int, default=10, help="Patch size of the ViT encoder.")
    g.add_argument("--depth", type=int, default=4, help="Transformer depth of the encoder.")
    g.add_argument("--dec_depth", type=int, default=4, help="Depth of the coordinate decoder.")
    g.add_argument("--heads", type=int, default=8, help="Attention heads.")
    g.add_argument("--n_fourier", type=int, default=64,
                   help="Random Fourier features of the coordinate embedding.")
    g.add_argument("--sigma", type=float, default=3.0,
                   help="Bandwidth of the random Fourier features.")
    g.add_argument("--n_query", type=int, default=8192,
                   help="RTE points (cell x direction) subsampled per step.")
    g.add_argument("--n_query_phys", type=int, default=2048,
                   help="Physics collocation points per step when --w_phys > 0. Kept "
                        "below --n_query on purpose: the residual needs a graph through "
                        "the cross-attention decoder, where 8192 points run out of memory.")
    g.add_argument("--val_every", type=int, default=25,
                   help="Validate every N epochs (full-angle validation is expensive).")
    g.add_argument("--pretrained_encoder", default="auto",
                   help="Encoder to start from: 'auto' (= --name), a .pt path, another "
                        "run name, or 'scratch' for a random encoder trained jointly.")
    g.add_argument("--data_frac", type=float, default=1.0,
                   help="Fraction of the TRAIN split to use (data-efficiency study). "
                        "The val split is never subsampled.")

    g = p.add_argument_group("encoder (stage 1)")
    g.add_argument("--encoder_epochs", type=int, default=1000,
                   help="Reconstruction epochs of the AE/Transformer-Based DeepONet "
                        "encoder. Ignored by pca (direct fit, no notion of epoch).")
    g.add_argument("--encoder_batch_size", type=int, default=64)
    g.add_argument("--encoder_pretrain_lr", type=float, default=1e-3,
                   help="LR of the encoder during its own reconstruction pretraining.")
    g.add_argument("--encoder_n_query", type=int, default=4096,
                   help="Transformer-Based DeepONet only: query points per step during "
                        "reconstruction pretraining.")
    g.add_argument("--subsample", default="auto",
                   help="Transformer-Based DeepONet only: random input subsampling "
                        "('auto', 'none', or e.g. '1,3,5').")
    g.add_argument("--latent_dim", type=int, default=500,
                   help="ae only: latent size of the convolutional autoencoder.")
    g.add_argument("--affine", action="store_true",
                   help="Drop the output softplus. For pca and ae the reconstructor is "
                        "then affine in beta -- exactly the class Theorem 3.4 bounds; "
                        "for tbd the cross-attention decoder stays nonlinear, and the "
                        "flag only isolates what the softplus itself buys.")
    g.add_argument("--pca_components", type=int, default=500,
                   help="pca only: number of PCA axes kept per input field.")
    g.add_argument("--pca_no_scaler", action="store_true",
                   help="pca only: raw PCA without StandardScaler (= POD).")
    g.add_argument("--retrain_encoder", action="store_true",
                   help="Refit the encoder even if its checkpoint already exists. By "
                        "default an existing encoder is reused, which is how several "
                        "--p_basis runs share one encoder.")

    g = p.add_argument_group("losses and bookkeeping")
    g.add_argument("--w_data", type=float, default=1.0)
    g.add_argument("--w_phys", type=float, default=0.0,
                   help="Weight of the RTE residual loss. 0 = data-only; the residual "
                        "is still monitored and plotted.")
    g.add_argument("--w_bc", type=float, default=0.0)
    g.add_argument("--loss_balancing", default="static", choices=["static", "relobralo"],
                   help="static = fixed w_*; relobralo = adaptive (w_* become on/off switches).")
    g.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Resume an interrupted run from its _resume.pt state.")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dataset is None:
        from utils import DEFAULT_DATASET
        args.dataset = DEFAULT_DATASET
    if args.batch_size is None:
        args.batch_size = 64 if args.model == "tbd" else 256
    if args.model == "pca" and not args.freeze_encoder:
        # A PCA basis has no trainable weights; the underlying CLI rejects the flag.
        args.freeze_encoder = True

    print(f"=== RTE2D train | model={args.model} | name={args.name} | "
          f"device={'CUDA' if torch.cuda.is_available() else 'CPU'} ===")

    if args.stage in ("all", "encoder"):
        from encoders import encoder_run
        ckpt = encoder_checkpoint(args.model, args.name)
        # `AE-p128-softplus` and `AE-p128-affine` differ by their output head alone and
        # share the encoder of the `AE-p128-softplus` configuration, so the encoder stage runs
        # under THAT name -- it is fitted once, and the second head reuses it.
        base = encoder_run(args.name)
        if os.path.exists(ckpt) and not args.retrain_encoder:
            print(f"[train.py] encoder already present, reused: {ckpt}"
                  + (f"\n           (shared with the other head of {base!r})"
                     if base != args.name else
                     "\n           (pass --retrain_encoder to refit it)"))
        else:
            if base != args.name:
                print(f"[train.py] {args.name!r} names an output head: its encoder is "
                      f"fitted once under {base!r} and shared by both heads.")
            enc_args = copy.copy(args)
            enc_args.name = base
            run_stage(enc_args, "encoder")

    if args.stage in ("all", "operator"):
        run_stage(args, "operator")
        print(f"\n[train.py] done. Evaluate with:\n"
              f"    python src/evaluate.py --name {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
