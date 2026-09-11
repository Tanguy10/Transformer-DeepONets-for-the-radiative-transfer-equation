# Transformer DeepONets for the radiative transfer equation

Code accompanying the paper *Beyond Affine Reconstruction: Multi-Resolution Transformer
DeepONets for the Radiative Transfer Equation*.

## The problem

We learn the solution operator of the steady radiative transfer equation (RTE) in a
two-dimensional, gray, absorbing and emitting medium, without scattering:

```math
\mathbf{s}(\varphi)\cdot\nabla I(\mathbf{x},\varphi) = \kappa(\mathbf{x})\,\big(I_b(\mathbf{x}) - I(\mathbf{x},\varphi)\big),
\qquad I_b = \frac{\sigma T^4}{\pi},
\qquad \mathbf{s}(\varphi) = (\sin\varphi,\ \cos\varphi),
```

on D = (−1.5, 1.5) × (0, 3) m, the angle φ ∈ [0, 2π) being measured clockwise from the y
axis. From the temperature T and the absorption coefficient κ, the models predict the
radiative intensity:

```math
\mathcal{G} : (T, \kappa) \longmapsto I.
```

The reference data are FVDOM solutions of a 2D pool fire (FireFOAM `smallPoolFire2D`):
6001 snapshots on a 150 × 150 mesh and 16 ordinates φ_m = 2πm/16. The networks are queried
pointwise at ξ = (x, y, φ); the incident radiation G = ∫ I dΩ ≈ (4π/16) Σ_m I(·, φ_m) is
evaluated alongside I.

The dataset was provided by Xiaoyi Lu (FM Global) and is the one used in X. Lu, M. Daneker,
L. Lu and Y. Wang, *Machine learning accelerated radiative transfer modeling in CFD fire
simulations*, Fire Safety Journal 163, 104819 (2026). It is not included in this repository.

Three models learn this operator:

- **PCA-DeepONet** / **AE-DeepONet** (`pca`, `ae`) — a PCA or a convolutional autoencoder per
  input field, two branch nets fused by a Hadamard product, and the affine branch·trunk
  reconstruction ⟨β, γ(ξ)⟩ + b of rank p;
- **Transformer-Based DeepONet** (`tbd`) — a Perceiver encoder over patches of (T, κ), which
  reads inputs at several resolutions without resampling, and a cross-attention
  reconstructor, not affine in the latent state.

## Repository structure

```
scripts/                     one driver per table / figure of the paper
├── train_paper_models.sh    trains every model of the paper
├── lower_bound.sh           lower bound of Theorem 3.4 and the models against it
├── power_spectrum.sh        power spectral density of G
├── multi_resolution.sh      mixed-resolution experiment
├── physics_loss.sh          RTE residual as a loss term
├── ablation_junction.sh     Perceiver encoder with an affine reconstruction
├── tbd_affine_head.sh       Transformer-Based DeepONet without output softplus
├── run_all.sh               all of the above, in order
└── _common.sh               shared helpers: logging, restart, GPU wait
src/
├── train.py                 training entry point (pca | ae | tbd)
├── evaluate.py              evaluation entry point (metrics + figures)
├── deeponet.py              PCA-/AE-DeepONet (branch·trunk, Hadamard product) + trainer
├── deeponet_joint.py        Transformer-Based DeepONet pipeline: CLI, data, checkpoints
├── tbdeeponet.py            Transformer-Based DeepONet model: encoder, reconstructor, training
├── encoders.py              PCA and convolutional autoencoder
├── utils.py                 data split, metrics, figures
├── prepare_data.py          builds the dataset from the raw solver output
├── lower_bound/             computable lower bound (Lanthaler et al., Thm 3.4) — bound.py
├── power_spectrum/          power spectral density of G and I
├── robustness_resolution/   training and evaluation on mixed 150² / 75² / 25² inputs
└── dot_decoder/             Perceiver encoder + branch·trunk junction
data/mesh_stub.h5            placeholder mesh (zeros), used when the raw H5 is absent
```

Checkpoints, logs and results are written to `src/checkpoints/`, `src/logs/` and
`src/results/<name>/` (not versioned). Design notes are in [`src/README.md`](src/README.md);
each sub-experiment directory documents itself.

## Training

Python 3.11, PyTorch ≥ 2.4 and one CUDA GPU. From the repository root:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
python src/prepare_data.py --h5 /path/to/smallPoolFire2D.h5   # -> data/dataset_train_full_phi.pt (~10 GB)
```

The 6001 snapshots are split once, with a fixed seed, into 4200 training / 900 validation /
901 test snapshots (70 / 15 / 15 %). One command trains a model:

```bash
python src/train.py --model pca --name PCA-p512-affine --p_basis 512 --affine
python src/train.py --model ae  --name AE-p128-softplus --p_basis 128
python src/train.py --model tbd --name TBD-N4-D128 -N 4 -D 128
```

Every run has two stages (`--stage encoder | operator | all`, default `all`):

1. **Encoder** — pretrained to reconstruct (T, κ): 1000 epochs, Adam, constant learning
   rate 1e-3, MSE loss. The PCA (500 axes per field) is a direct fit; the autoencoder has a
   latent size of 500 per field.
2. **Operator** — trained to predict I: 2000 epochs, Adam, learning rate 1e-3 decayed by a
   factor 0.9977 per epoch. The AE and Transformer encoders are fine-tuned jointly at 1e-4
   under the same decay; the PCA stays fixed. Loss: per-snapshot relative L2 error on I.
   Batches of 256 snapshots (PCA/AE) or 64 (Transformer). The checkpoint with the best
   validation error is kept.

Main options: `--p_basis` rank p of the junction (PCA/AE, default 128), `--affine` removes
the output softplus (PCA/AE), `-N` / `-D` number and width of the latent tokens
(Transformer, default 4 / 128), `--freeze_encoder` keeps the pretrained encoder fixed,
`--w_phys` adds the RTE residual to the loss (default 0). `python src/train.py --help` lists
them all. An interrupted run resumes when the same command is relaunched.
`bash scripts/train_paper_models.sh` trains every model of the paper.

## Evaluation

```bash
python src/evaluate.py --name TBD-N4-D128                     # test split
python src/evaluate.py --name TBD-N4-D128 --split test,train  # several splits
```

The architecture is read back from the checkpoint. The script reports the per-snapshot
relative L2 error on I and on G over the split, and plots G (reference, prediction, error)
for a few snapshots, into `src/results/<name>/`. `--recon` adds the reconstruction error of
the encoder alone, `--physics` the RTE residual of the prediction, and
`--encoders pca:<run>,ae:<run>` compares encoders without any operator.

`bash scripts/lower_bound.sh` computes the lower bound of Theorem 3.4 on the training and
test sets and places the trained models against it; the method is
[`src/lower_bound/bound.py`](src/lower_bound/bound.py), its derivation
[`src/lower_bound/Lower_Bound.md`](src/lower_bound/Lower_Bound.md).
