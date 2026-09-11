# `robustness_resolution/` — one dataset, three input resolutions

**Table III and Figures 8–11 of the paper.** The question: a surrogate standing in for a
solver is called on whatever mesh the simulation uses. What does that cost each family?

From the reference dataset we build a **mixed-resolution** one by block averaging — a
third of the snapshots presented at 150², a third at 75², a third at 25² (Fig. 8). The
averaging is applied to T, κ and I alike, so a snapshot presented at 25² carries its input
pair on the 25×25 mesh and its intensity at the corresponding 25²×16 query points; the 16
ordinates are never resampled. The split of the parent repository is kept, the resolution
of a snapshot is drawn once, and every model sees exactly the same data.

The five models trained on it:

| run | family | how it reads a coarse input |
|---|---|---|
| `res-TBD` | Transformer-Based DeepONet | **natively** — the patch sequence lengthens, nothing learned changes size |
| `res-PCA-150` | PCA-DeepONet | every input interpolated **up** to the 150² solver mesh |
| `res-PCA-50` | PCA-DeepONet | everything brought **down** to a 50² working grid |
| `res-AE-150` | AE-DeepONet | up to 150² |
| `res-AE-50` | AE-DeepONet | down to 50² |

The four fixed-dimension models carry the **affine head** (`--affine`, no output
softplus), so their reconstructor is literally the R of Theorem 3.4 — the class the
attention decoder is compared against. Their PCA keeps 500 axes per field, as in the main
campaign.

At test time the input pair is supplied at each of the three resolutions in turn, but the
prediction is **always requested at the 150²×16 query points of the solver mesh** and
compared with the field stored there. The reference is thus the same throughout, and what
varies across the columns of Table III is only how much of the input the encoder was
given.

## Running it

```bash
setsid nohup bash scripts/multi_resolution.sh \
    > src/robustness_resolution/logs/campaign.log 2>&1 &
```

About 15 h on a single GPU. Restartable: one `.done_<tag>` marker per step, relaunch the
same file after a cut. Derived datasets are rebuilt on first use next to the main one
(`mixed_res_dataset_phi.pt`, `interp150_dataset_phi.pt`, `fixedres50_dataset_phi.pt`);
`RTE2D_DERIVED_DATA_DIR` moves them elsewhere.

Individual steps, if you only want one:

```bash
export PYTHONPATH=$PWD/src
python src/robustness_resolution/src/train.py --model tbd
python src/robustness_resolution/src/train.py --model pca --affine --input_res 50
python src/robustness_resolution/src/evaluate.py --split test --test mixed --tag mixed \
       --model tbd:res-TBD --model pca:res-PCA-150 --model ae:res-AE-150 \
       --model pca:res-PCA-50:50 --model ae:res-AE-50:50
python src/robustness_resolution/src/evaluate.py --fields 3155      # Figure 8
```

`--input_res` also accepts resolutions **no model was trained on** (30, 100, 125…), which
is how the generalisation comb is measured.

## Caveat

`checkpoints/encoder_tbd_pretrained.pt`, the frozen encoder of `res-TBD`, was pretrained
under the **historical 85/15 split**: its reconstruction stage saw the (T, κ) fields of
today's test block. It never saw an intensity and never saw a label, so the supervised
stage is clean, but the encoder is not strictly blind to the test inputs. Retraining it
under 70/15/15 costs several GPU hours.
