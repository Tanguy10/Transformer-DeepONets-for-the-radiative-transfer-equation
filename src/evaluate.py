#!/usr/bin/env python
import argparse
import json
import os
import sys

import torch

FAMILIES = ("pca", "ae", "tbd")
ENCODER_KIND = {"pca": "pca", "ae": "autoencoder", "tbd": "tbd"}
ENCODER_LABEL = {"pca": "PCA", "autoencoder": "AE",
                 "tbd": "TB-DeepONet", "joint_ae": "AE (joint)"}


def detect_model(name):
    """Guess the family of `name` from the checkpoints present on disk."""
    from deeponet_joint import operator_path
    from encoders import encoder_path
    from utils import CHECKPOINTS_DIR

    found = []
    if os.path.exists(operator_path(name)):
        found.append("tbd")
    if os.path.exists(os.path.join(CHECKPOINTS_DIR, f"{name}.pth")):
        if os.path.exists(encoder_path(name, "pca")):
            found.append("pca")
        if os.path.exists(encoder_path(name, "autoencoder")):
            found.append("ae")
    if not found:
        raise FileNotFoundError(
            f"No checkpoint found for run {name!r} in {CHECKPOINTS_DIR}.\n"
            f"Train it first, e.g.: python src/train.py --model pca --name {name}")
    if len(found) > 1:
        raise ValueError(
            f"Run {name!r} matches several families ({', '.join(found)}). "
            f"Disambiguate with --model.")
    return found[0]


def load_pair(name, model, device):
    """(network, encoder) of a run, whatever its family."""
    if model == "tbd":
        from deeponet_joint import operator_path
        from tbdeeponet import load_operator
        return load_operator(operator_path(name), device=device)
    from utils import load_any_encoder, load_any_model
    cfg = {"name": name, "encoder": ENCODER_KIND[model], "arch": "modular"}
    return load_any_model(cfg, device), load_any_encoder(cfg, device)


def _eval_argv(args, model, split, snapshots):
    common = ["--name", args.name, "--dataset", args.dataset,
              "--eval_split", split, "--eval_snapshots", snapshots,
              "--batch_size_data", str(args.batch_size),
              "--g_ref", args.g_ref,
              "--output_dir", args.output_dir]
    if model == "tbd":
        return ["--mode", "eval_operator"] + common
    return ["--mode", "eval_2d", "--encoder", ENCODER_KIND[model]] + common


def run_physics(name, model, splits, split, device, n_cells, n_dirs, seed,
                n_snapshots=None, chunk=None):
    """RTE residual of the prediction on `split`; `chunk` = snapshots per pass."""
    from utils import build_collocation, evaluate, report, split_data
    import math

    net, encoder = load_pair(name, model, device)
    data_eval = split_data(splits, split)
    i_max = float(data_eval["I_target"].max())
    y, cell_idx, T, kappa = build_collocation(
        data_eval, splits["y_full"], n_dirs, n_cells,
        trained_dirs=False, seed=seed, device=device)
    acc = evaluate(net, encoder, y, cell_idx, T, kappa, device, seed=seed,
                   n_snapshots=n_snapshots, chunk=chunk or (2 if model == "tbd" else 64))
    rho, corr = report(acc, f"{name}_{split}", i_max=i_max)
    return {"rho": rho, "corr": corr,
            "res_imax": math.sqrt(acc["r2"] / acc["n"]) / i_max}


def run_recon(name, model, splits, split, snapshots, output_dir, device, prefer_ft):
    """Reconstruction quality of the encoder alone: metric over the split + figures."""
    from utils import reconstruction_report
    # The fine-tuned TBD encoder lives in the operator checkpoint, not in an _ft file.
    encoder = load_pair(name, model, device)[1] if (prefer_ft and model == "tbd") else None
    stats = reconstruction_report(
        ENCODER_KIND[model], name, split=split, aggregate=True, snapshots=snapshots,
        output_dir=output_dir, device=device, prefer_ft=prefer_ft, splits=splits,
        encoder=encoder)
    out = {}
    for s, key in zip(stats, ("recon_T", "recon_kappa")):
        out[key] = s["mean"]
    return out


def parse_encoder_spec(spec):
    """'pca:my-run', 'ae:my-run', 'tbd:my-run' -- or just 'my-run', whose kind is then
    read off the encoder checkpoints present on disk."""
    from deeponet_joint import joint_encoder_path
    from encoders import encoder_path

    alias = {"pca": "pca", "ae": "autoencoder", "autoencoder": "autoencoder",
             "tbd": "tbd", "joint_ae": "joint_ae"}
    if ":" in spec:
        kind, run = spec.split(":", 1)
        if kind not in alias:
            raise SystemExit(f"unknown encoder kind {kind!r} in {spec!r} "
                             f"({' | '.join(sorted(alias))})")
        return alias[kind], run

    run = spec
    found = [k for k in ("pca", "autoencoder")
             if os.path.exists(encoder_path(run, k))]
    found += [k for k, tag in (("tbd", "tbd"), ("joint_ae", "ae"))
              if os.path.exists(joint_encoder_path(run, tag))]
    if not found:
        raise SystemExit(f"No encoder checkpoint found for {run!r}. Train one first, "
                         f"e.g.: python src/train.py --model pca --name {run}")
    if len(found) > 1:
        raise SystemExit(f"Run {run!r} owns several encoders ({', '.join(found)}); "
                         f"prefix the kind, e.g. --encoders {found[0]}:{run}")
    return found[0], run


def run_encoders(args, splits, split, snapshots, device):
    """Reconstruction error of each --encoders entry over the split, plus figures."""
    from utils import (encoder_split_error, load_any_encoder, plot_encoder_comparison,
                       resolve_snapshots, split_data)

    data = split_data(splits, split)
    entries, records = [], []
    for spec in [s.strip() for s in args.encoders.split(",") if s.strip()]:
        kind, run = parse_encoder_spec(spec)
        encoder = load_any_encoder({"name": run, "encoder": kind}, device,
                                   prefer_ft=args.recon_ft)
        d = encoder.latent_dims
        dim = f"{d['u1_in']}" if d["u1_in"] == d["u2_in"] \
            else f"{d['u1_in']} / {d['u2_in']}"
        label = f"{ENCODER_LABEL.get(kind, kind)}, d = {dim}"
        entries.append((label, encoder))
        stats = encoder_split_error(encoder, data, kind, split=split)
        records.append({"run": run, "kind": kind, "label": label,
                        "latent_T": int(stats[0]["latent_d"]),
                        "latent_kappa": int(stats[1]["latent_d"]),
                        "T": {k: float(stats[0][k]) for k in ("mean", "std", "max")},
                        "kappa": {k: float(stats[1][k]) for k in ("mean", "std", "max")}})

    figures = []
    if snapshots:
        raw = splits[f"{split}_idx"]
        for snap in resolve_snapshots(snapshots, data):
            figures.append(plot_encoder_comparison(
                entries, data, snap, output_dir=args.output_dir,
                raw_idx=int(raw[snap]), split=split))
    return records, int(data["u1_in"].shape[0]), figures


def rel_to_root(path):
    """Path relative to the repository, so that the files written stay portable."""
    from utils import ROOT
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def write_encoder_metrics(records, args, split, n_snapshots, output_dir):
    """Persist the encoder-only reconstruction numbers, as text and as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    record = {"name": args.name, "split": split, "dataset": rel_to_root(args.dataset),
              "n_snapshots": n_snapshots, "encoders": records}

    json_path = os.path.join(output_dir, f"{args.name}_recon_{split}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    txt_path = os.path.join(output_dir, f"{args.name}_recon_{split}.txt")
    with open(txt_path, "w") as f:
        f.write(f"split   : {split}  ({n_snapshots} snapshots)\n")
        f.write(f"dataset : {rel_to_root(args.dataset)}\n")
        f.write(f"encoder : as trained{' (fine-tuned)' if args.recon_ft else ''}\n\n")
        f.write("Encoder alone, reconstruction relative L2 per snapshot "
                "(mean over the split)\n")
        for r in records:
            f.write(f"\n  {r['run']}  ({r['kind']}, d = {r['latent_T']} / "
                    f"{r['latent_kappa']})\n")
            for key, name in (("T", "T    "), ("kappa", "kappa")):
                s = r[key]
                f.write(f"    {name} : {100*s['mean']:6.2f} %   "
                        f"(std {100*s['std']:5.2f}  max {100*s['max']:6.2f})\n")
    print(f"Metrics written to {txt_path} and {json_path}")
    return record


def write_metrics(metrics, args, model, split, output_dir):
    """Persist the metrics next to the figures, as text and as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    cast = lambda k, v: None if v is None else (int(v) if k.startswith("n_") else float(v))
    record = {"name": args.name, "model": model, "split": split,
              "dataset": rel_to_root(args.dataset),
              **{k: cast(k, v) for k, v in metrics.items()}}

    json_path = os.path.join(output_dir, f"{args.name}_metrics_{split}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    txt_path = os.path.join(output_dir, f"{args.name}_metrics_{split}.txt")
    pct = lambda k: 100 * metrics[k]
    with open(txt_path, "w") as f:
        f.write(f"model   : {model}\n")
        f.write(f"run     : {args.name}\n")
        f.write(f"split   : {split}  ({int(metrics['n_snapshots'])} snapshots)\n")
        f.write(f"dataset : {rel_to_root(args.dataset)}\n\n")
        f.write("Per-snapshot RELATIVE errors (mean over the split)\n")
        f.write(f"  I  L2 : {pct('I_l2'):6.2f} %   "
                f"(std {pct('I_l2_std'):5.2f}  min {pct('I_l2_min'):5.2f}  "
                f"max {pct('I_l2_max'):6.2f})\n")
        f.write(f"  G  L2 : {pct('G_l2'):6.2f} %   "
                f"(std {pct('G_l2_std'):5.2f}  min {pct('G_l2_min'):5.2f}  "
                f"max {pct('G_l2_max'):6.2f})\n")
        f.write(f"  I  L1 : {pct('I_l1'):6.2f} %\n")
        f.write(f"  G  L1 : {pct('G_l1'):6.2f} %\n\n")
        f.write("ABSOLUTE errors, physical units (I in W/m2/sr, G in W/m2)\n")
        f.write(f"  I  E_hat = RMS_m ||I - N(u)||  : {metrics['I_ehat']:.4e}   "
                f"RMSE/point {metrics['I_rmse']:.4e}   MAE {metrics['I_mae']:.4e}\n")
        f.write(f"  G  E_hat = RMS_m ||G - G_pred||: {metrics['G_ehat']:.4e}   "
                f"RMSE/point {metrics['G_rmse']:.4e}   MAE {metrics['G_mae']:.4e}\n")
        if metrics.get("recon_T") is not None:
            f.write("\nEncoder reconstruction (encoder alone, relative L2 per snapshot)\n")
            f.write(f"  T     : {pct('recon_T'):6.2f} %\n")
            f.write(f"  kappa : {pct('recon_kappa'):6.2f} %\n")
        if metrics.get("rho") is not None:
            f.write("\nRTE residual r = mu.grad(I) + kappa*I - S\n")
            f.write(f"  rho = RMS(r)/RMS(S) : {pct('rho'):6.2f} %\n")
            f.write(f"  RMS(r)/max(I)       : {pct('res_imax'):6.2f} %\n")
            f.write(f"  corr(transport, S - kappa*I) : {metrics['corr']:+.3f}\n")
    print(f"Metrics written to {txt_path} and {json_path}")
    return record


def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model: split metrics, snapshot figures, "
                    "encoder reconstruction, RTE residual.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", default=None,
                   help="Run name given to train.py (optional with --encoders, where it "
                        "only names the output files; default 'encoders').")
    p.add_argument("--encoders", default=None,
                   help="ENCODER-ONLY mode, no DeepONet needed: comma separated list of "
                        "encoders to judge on their reconstruction alone, as "
                        "'kind:run' ('pca:PCA-p128,ae:AE-p128' -- the encoder of a "
                        "configuration, shared by both of its heads) or plain 'run' "
                        "when a single encoder checkpoint carries that name. Writes the "
                        "relative L2 of T and kappa over --split and one comparison "
                        "figure per --recon_snapshots snapshot (rows T/kappa, columns "
                        "truth then each encoder).")
    p.add_argument("--model", default="auto", choices=("auto",) + FAMILIES,
                   help="Model family. 'auto' reads it off the checkpoints on disk.")
    p.add_argument("--g_ref", default="integrated", choices=("integrated", "solver"),
                   help="Reference for G: the integrated sum_m I_target (default), or the "
                        "solver's own G field read from RTE2D_SOLVER_H5. The two agree to "
                        "0.88 eps(float32), so this changes no number.")
    p.add_argument("--split", default="test",
                   help="Splits to measure, comma separated among train/val/test "
                        "(e.g. 'test,train'). Figures are written for the FIRST one.")
    p.add_argument("--snapshots", default=None,
                   help="Snapshots to plot, comma separated: direct indices ('0,20'), "
                        "-1 = the most intense one, 'structure:k' = the k most "
                        "structured (the default), 'chaotic:k' = the k of largest spatial "
                        "variance of G, 'random:k' = k drawn at a fixed seed, hence "
                        "identical across models. 'none' disables the figures.")
    p.add_argument("--recon", action="store_true",
                   help="Also measure what the encoder alone rebuilds (T and kappa "
                        "relative L2 over the split) and plot --recon_snapshots of them.")
    p.add_argument("--recon_snapshots", default="chaotic:1,random:2",
                   help="Snapshots of the reconstruction figures (same syntax as "
                        "--snapshots); 'none' keeps the metric without the figures.")
    p.add_argument("--recon_ft", action="store_true",
                   help="Judge the FINE-TUNED encoder (the weights the operator uses) "
                        "instead of the pretrained one.")
    p.add_argument("--physics", action="store_true",
                   help="Also measure the RTE residual of the prediction (rho, "
                        "RMS(r)/max(I), transport correlation).")
    p.add_argument("--phys_cells", type=int, default=4000,
                   help="Cells drawn for the residual collocation.")
    p.add_argument("--phys_dirs", type=int, default=16,
                   help="Continuous phi directions per cell for the residual.")
    p.add_argument("--phys_chunk", type=int, default=0,
                   help="Snapshots differentiated at once for the residual "
                        "(0 = 64 for pca/ae, 2 for tbd). Lower it on an OOM.")
    p.add_argument("--phys_snapshots", type=int, default=0,
                   help="Snapshots used for the residual (0 = the whole split). The knob "
                        "to cut its cost: it is the only part that back-propagates.")
    p.add_argument("--dataset", default=None,
                   help="Path to the .pt dataset (default: data/dataset_train_full_phi.pt)")
    p.add_argument("--output_dir", default=None, help="Default: results/<name>/")
    p.add_argument("--batch_size", type=int, default=256, help="Snapshots per eval batch.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    from utils import DEFAULT_DATASET, RESULTS_DIR, SNAPSHOT_SPEC_DEFAULT
    if args.name is None:
        if not args.encoders:
            raise SystemExit("--name is required (or --encoders for the encoder-only mode)")
        args.name = "encoders"
    if args.dataset is None:
        args.dataset = DEFAULT_DATASET
    if args.snapshots is None:
        args.snapshots = SNAPSHOT_SPEC_DEFAULT
    if args.output_dir is None:
        args.output_dir = os.path.join(RESULTS_DIR, args.name)

    splits_wanted = [s.strip() for s in args.split.split(",") if s.strip()]
    unknown = [s for s in splits_wanted if s not in ("train", "val", "test")]
    if unknown:
        raise SystemExit(f"unknown split(s): {', '.join(unknown)} (train | val | test)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.encoders:
        from utils import load_and_split_dataset
        print(f"=== RTE2D evaluate | encoders={args.encoders} | "
              f"splits={','.join(splits_wanted)} | device={device.type.upper()} ===")
        splits = load_and_split_dataset(args.dataset, map_location="cpu")
        for k, split in enumerate(splits_wanted):
            snaps = None if (k > 0 or args.recon_snapshots.lower() == "none") \
                else args.recon_snapshots
            records, n_snap, _ = run_encoders(args, splits, split, snaps, device)
            write_encoder_metrics(records, args, split, n_snap, args.output_dir)
            print(f"\n=== encoder reconstruction | {split} ({n_snap} snapshots) ===")
            for r in records:
                print(f"  {r['run']:24s} {r['kind']:12s} "
                      f"T {100*r['T']['mean']:6.2f} %   "
                      f"kappa {100*r['kappa']['mean']:6.2f} %")
        print(f"Figures written to {args.output_dir}")
        return 0

    model = detect_model(args.name) if args.model == "auto" else args.model
    print(f"=== RTE2D evaluate | model={model} | name={args.name} | "
          f"splits={','.join(splits_wanted)} | device={device.type.upper()} ===")

    if model == "tbd":
        import deeponet_joint as backend
        run_eval = backend.run_eval_operator
    else:
        import deeponet as backend
        run_eval = backend.run_eval_2d

    splits = None
    records = []
    for k, split in enumerate(splits_wanted):
        # First split only: figure names carry no split, a second would overwrite them.
        snapshots = args.snapshots if k == 0 else "none"
        snapshots = "" if snapshots.lower() == "none" else snapshots

        sub = backend.build_parser().parse_args(_eval_argv(args, model, split, snapshots))
        metrics = run_eval(sub, device)

        if args.recon or args.physics:
            from utils import load_and_split_dataset
            if splits is None:
                splits = load_and_split_dataset(args.dataset, map_location="cpu")
        if args.recon:
            recon_snaps = None if (k > 0 or args.recon_snapshots.lower() == "none") \
                else args.recon_snapshots
            metrics.update(run_recon(args.name, model, splits, split, recon_snaps,
                                     args.output_dir, device, args.recon_ft))
        if args.physics:
            metrics.update(run_physics(args.name, model, splits, split, device,
                                       args.phys_cells, args.phys_dirs, seed=0,
                                       n_snapshots=args.phys_snapshots or None,
                                       chunk=args.phys_chunk or None))

        records.append(write_metrics(metrics, args, model, split, args.output_dir))

    print(f"\n=== {args.name} ({model}) ===")
    for r in records:
        line = (f"  {r['split']:5s}  I-L2 {100*r['I_l2']:5.2f} %  G-L2 {100*r['G_l2']:5.2f} %"
                f"  |  I E_hat {r['I_ehat']:.3e}  G E_hat {r['G_ehat']:.3e}")
        if r.get("recon_T") is not None:
            line += f"  |  recon T {100*r['recon_T']:.2f} % kappa {100*r['recon_kappa']:.2f} %"
        if r.get("rho") is not None:
            line += f"  |  rho {100*r['rho']:.1f} %"
        print(line)
    print(f"Figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
