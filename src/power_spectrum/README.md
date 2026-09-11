# `power_spectrum/` — PSD: FVDOM truth against the models

One question only: **down to what spatial scale do the models reproduce the field?**

We plot the power spectral density — the energy carried by each spatial scale — for the
FVDOM truth and for the repository's main models.

This is what a global L2 error cannot see: L2 is **dominated by the large scales**, which
carry most of the energy. A model can post a 5 % L2 error while having *strictly no* fine
structure at all; the PSD shows it, L2 does not.

## The two families of figures

| figure | quantity | how many |
|---|---|---|
| `psd_G.pdf` | `G = ∫ I dφ`, the angular integral | one |
| `psd_I_phi<deg>.pdf` | `I(x, y, φ_m)` in **one single** direction | one per ordinate (16) |

Both share exactly the **same layout**: one panel per source, in the order of
`--sources`. Each panel carries

* the **individual snapshots** (thin lines, 120 drawn at random by default, `--n-show`)
  — the spread that the mean erases;
* their **mean** over the whole block (thick coloured line), computed over all 901
  snapshots and not only the plotted ones;
* the **mean of the truth for the same quantity** (black dashes), repeated in every
  panel — so a model reads against its reference without leaving its panel, and the
  comparison always stays within one direction.

**Shared axes**, without which every panel would be re-stretched and all of them would
end up looking alike by construction. The scale is shared by the panels of a figure **and
by the 16 angle figures**: going from `psd_I_phi270.pdf` (where the models stick to the
truth) to `psd_I_phi112.pdf` (where they fall a decade short) directly shows that the
deficit is not spread uniformly over the directions. The same snapshots are drawn in
every figure.

What the terminal summarises at the end of each run, per model: the L2 error on `I` and
on `G` (a checkpoint sanity check), then the **tail energy** — the model/truth energy
ratio integrated over `k ∈ [40, 74]`, i.e. structures of 4 cells and below — for `G`, and
its range over the 16 directions.

## The sources

All interchangeable as far as `run.py` is concerned: each produces a flattened `I` of
shape `(B, 360000)`, then goes through exactly the same spectral treatment.

| key | model | checkpoint | loading |
|---|---|---|---|
| `truth` | FVDOM solver | `I_target` of the `.pt` | — |
| `pca` | PCA-DeepONet | `PCA-p512-softplus.pth` | `utils.load_any_model` / `load_any_encoder` |
| `ae` | AE-DeepONet | `AE-p512-softplus.pth` | idem |
| `tbd` | Transformer-Based DeepONet | `operator_tbd_TBD-N4-D128.pt` | `tbdeeponet.load_operator` |

The DeepONets consume the latent pair `(z_T, z_κ)`; the Transformer-Based DeepONet only
reads `branch_inputs[0]` (the joint latent) and ignores the second — so the same
`model(latents, coords)` signature works for both families, hence a single
`fields.predict_I` path.

The three model keys are the best run of each family in Table II of the paper, at the
same p and the same 500-component branch input for `pca` and `ae`: what separates their
two panels is the encoder, nothing else.

### Any other model

The keys above are shortcuts to the runs of the repository's reference table. **Any run
trained by `train.py` is a panel of its own**, simply named on the command line:

```bash
python src/power_spectrum/run.py --sources truth tbd my-run
```

Its family is read off the checkpoints on disk, as `evaluate.py --name` does; prefix it
(`tbd:my-run`, `pca:my-run`, `ae:my-run`) only if a single name carries several. The run
name then keys everything it produces — its cache `cache/<split>_<run>_n<N>.npz` and its
figures `psd_G_<run>.pdf` — so a model of yours never overwrites the reference figures:
`--tag` defaults to the first run named. The model must have been trained on the same
`(x, y, phi)` dataset; a 4-column checkpoint from the historical code base cannot be
queried with it.

The **test** block is the default (901 snapshots), 70/15/15 split with `seed=42`.
`--split val` for the validation block.

**Checkpoint sanity check**: the L2 error printed at the end of a run must land on the
repository's known figures. If it does not, it is the wrong checkpoint and the spectra
are worthless.

## The method

That of **Bert Vandenbroucke's post (2019)**, entirely recoded in [`psd.py`](psd.py):

> https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/

1. 2D FFT of the field — `F = np.fft.fftn(image)`
2. squared amplitudes — `A = |F|²`
3. wavevector norm, in integer mode indices — `k = √(kx² + ky²)`
4. **mean of `A` over rings** `k ∈ [n−0.5, n+0.5[`, converted back to a sum

No spectrum library: numpy alone (scipy only in `reference_power_spectrum`, the
self-test's control). The module is called `psd` rather than `spectrum` so it does not
shadow the PyPI package of that name — which would not fit here anyway, as it deals with
**1D** series and has no 2D ring averaging.

**One single departure from the post**: its step 4 multiplies the ring mean by the ring
**area** `π(k_ext²−k_int²)`, a continuous approximation of the **number** of integer
modes that fall in it. The two differ by −21 % on the first ring and by ±10 % afterwards
(RMS 7 %), in a sawtooth pattern, and that bias is **multiplicative and identical for
every snapshot**: averaging does not attenuate it. We use the exact count, which makes
Parseval exact to the bit. Removing the spatial mean and the `|F|²/npix⁴` normalisation
(Parseval: `Σ_k P(k)` = the variance of the field) complete the shaping.

⚠ **No apodisation.** The FFT assumes a periodic field; `I` is not (hot wall at the
bottom, cold at the top). The step at the edges is a broadband signal that inflates the
tail of the spectrum. Since this leakage adds to the truth as much as to the models, it
**pulls every ratio towards 1** and makes the tail band the less discriminating the more
it is aggregated: on `I` averaged over the 16 directions it saturates, whereas direction
by direction the gap stays readable (61–99 % depending on `φ` for the PCA-DeepONet).
Hence the choice to plot only `G` and the directions **separately**, never the angular
aggregate.

### Mean of the curves, not curve of the mean

We compute the PSD of **each** snapshot and then average the curves. This is **not** the
PSD of the mean field: since `⟨|F|²⟩ = |⟨F⟩|² + Var F`, that one would retain only the
**permanent** structure and lose all the turbulence. A model that predicted the mean
field perfectly with no fluctuation whatsoever would have a perfect
PSD-of-the-mean-field while being useless.

### How far k goes

`k` is not a plotting knob: it is the index of a DFT mode. On a 150×150 grid there are
exactly 22 500 of them, and the largest has `|k| = √2·75 = 106.07`. There is nothing
beyond — asking for `P(1000)` would be asking for a coefficient that is not in the array.
Three stopping points, exposed by `--kmax`:

| `--kmax` | final k | what it is |
|---|---|---|
| `complete` | 74 | last **complete** ring |
| **`nyquist`** (default) | **75** | `npix/2` — ring 75 only keeps 90 % of its modes |
| `corner` | 106 | last populated ring — the real end of the data, but there the ring spills out of the sampled square and survives only through its portions near the **diagonals**: the modes stay valid (the self-test checks it), their angular coverage collapses |

The cache is **always** computed out to the corners and then sliced: one file serves any
`--kmax`.

### The self-test

`psd.py` runs on its own, in ~2 s:

```
python src/power_spectrum/psd.py
  [ok] identical to the reference implementation (max relative gap).. 1.79e-15
  [ok] Parseval EXACT with shell='count' (relative gap)............... 4.09e-16
  [ok] Parseval with the reference ring area (gap).................... 2.27e-03
  [ok]   ... area/mode-count bias: 1st ring | RMS.................... -21 % | 6.9 %
  [ok] pure mode n=12: ring + amplitude A²/2.......................... 7.89e-16
  [ok] corner mode |k|=99 (> Nyquist 75): ring + A²/2................. 1.38e-14
  [ok] kmax='corner': exact Parseval on an arbitrary field............ 2.19e-16
  [ok]   ... power discarded by stopping at the disc (white noise).... 22.5 %
```

1. the vectorised version **reproduces the post to the bit** (same options);
2. **Parseval**: `Σ_k P(k)` = the spatial variance;
3. a **pure mode** `A·cos(2πnx/L)` falls in ring `n` with amplitude `A²/2`;
4. a **corner mode** of norm 99, well "beyond Nyquist", falls there just as exactly —
   proof that those modes are sampled, not aliased;
5. with `kmax='corner'` **no mode is lost**: Parseval holds on an arbitrary field,
   whereas stopping at the disc throws away nearly a quarter.

It runs at the start of `run.py` (`--no-selftest` to skip it).

## Files

```
psd.py       the method, dataset independent:
               reference_power_spectrum()  the post's code, verbatim (control)
               SpectrumBinner              vectorised version over a batch of images
               selftest()                  the 5 numerical checks
fields.py    mesh geometry, angles of the 16 ordinates, shaping into images,
             G = ∫I dφ, and supply of I: truth (I_target) or a model prediction.
             Checkpoint loading, single inference path. No physics, no spectral method.
figures.py   the panel shared by both families, the colours and the style
run.py       CLI: computation + cache + figures + sanity summary
cache/       {block}_{source}_n{N}.npz — `P_G` (N, 106) and `P_I` (N, 16, 106), the PSD
             of EVERY snapshot and EVERY direction, plus the per-snapshot L2 errors.
             Nothing is aggregated there: any statistic can be recomputed without
             rerunning inference. Always pushed out to the corners then sliced according
             to --kmax; N is in the filename, so a cache from another N cannot be reused
             by mistake.
results/     psd_G.pdf + the 16 psd_I_phi<deg>.pdf, suffixed by --tag (which
             defaults to the first run named in --sources)
```

The name of an angle figure carries the **integer part** of `φ` in degrees
(`psd_I_phi112.pdf` ⇒ `φ = 112.5°`, the exact angle being in the title): the 16 ordinates
being multiples of 22.5°, no collision is possible and alphabetical order is angle order.

## Running (from `RTE2D/`)

```bash
export PYTHONPATH=$PWD/src

python -u src/power_spectrum/psd.py                        # self-test only, ~2 s
python -u src/power_spectrum/run.py                        # everything, ~4 min cold
python -u src/power_spectrum/run.py --figures G            # the G figure only
python -u src/power_spectrum/run.py --angles 0 90 180 270  # 4 directions
python -u src/power_spectrum/run.py --n-snapshots 100      # quick look
python -u src/power_spectrum/run.py --split val            # validation block
python -u src/power_spectrum/run.py --refresh tbd          # recompute one model
python -u src/power_spectrum/run.py --xscale log           # logarithmic abscissa

# a model of your own, against the reference it should be read against
python -u src/power_spectrum/run.py --sources truth tbd TBD-N4-D128-phys
```

The cost is dominated by Transformer-Based DeepONet inference (~0.23 s/snapshot against
~0.02 s for the DeepONets: an attention decoder over 360 000 points). Everything is cached
— the figures regenerate in ~20 s. `--pts-per-fwd` reduces the GPU footprint when another
job is on the card.

⚠ `run.py` must be launched **from `RTE2D/`**: it imports `utils`, `deeponet`, `encoders`
and `tbdeeponet` from the parent package. In the background, give the script's **absolute** path
— a `setsid nohup` does not necessarily keep the working directory.

## Notes

* The models are evaluated on the **test** block, never seen during training: a tail
  deficit is not overfitting, it is a capacity or regularisation limit.
* The 16 directions are treated as 16 independent 2D images. The PSD therefore says
  nothing about the **angular** regularity of `I`, nor about untrained angles.
* No H5 file is required: everything comes from the `.pt` and the checkpoints.
* For the PSD of one particular snapshot, everything is already in the cache:

  ```python
  import numpy as np
  d = np.load("src/power_spectrum/cache/test_tbd_n901.npz")
  k = d["kvals"]
  psd_G_snap0 = d["P_G"][0]            # the G curve of snapshot 0
  psd_dir3 = d["P_I"][:, 3].mean(0)    # direction 3 (dataset order, unsorted)
  ```
