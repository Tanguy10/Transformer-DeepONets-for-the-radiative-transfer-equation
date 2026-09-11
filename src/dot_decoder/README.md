# `dot_decoder/` — what the cross-attention reconstructor actually buys

**Appendix C of the paper.** The Transformer-Based DeepONet differs from the baselines in
two places at once: it reads the input with a patch encoder, and it reconstructs with
cross-attention. Table II cannot tell the two apart. This sub-experiment holds the
encoder fixed and changes only the reconstructor.

Every run keeps the encoder of `TBD-N1-D128` and replaces its cross-attention decoder by
the branch·trunk junction of a vanilla DeepONet,

```
  I(u)(y) = act( ⟨z(u), t(y)⟩ + b ) · I_scale
```

where `z(u)` is the flattened latent (p = N·D) and `t(y)` the trunk output. Everything
else — pretraining, fine-tuning, budget, loss, schedule — is that of the reference run,
so the difference with `TBD-N1-D128` in Table II is the junction and nothing else.

| run | branch A: z → β | trunk t(y) | head |
|---|---|---|---|
| `abl-dot` | none, z **is** the coefficient vector | MLP on random Fourier features | softplus |
| `abl-dot-notrunk` | none | none — t(y) **is** the RFF projection | softplus |
| `abl-dot-branch` | MLP [512]×4 | MLP on RFF | softplus |
| `abl-dot-branch-raw` | MLP [512]×4 | MLP on the raw (x, y, φ) | softplus |
| `abl-dot-branch-affine` | MLP [512]×4 | MLP on RFF | **affine** — literally the R of Thm 3.4 |

Because these reconstructors are affine in the coefficient vector, the bound of
`lower_bound/` constrains them, and `--mode eval` places each one against it. The last
row is the only one covered to the letter, which is what makes it the honest comparison
point for the attention decoder.

## Running it

```bash
setsid nohup bash scripts/ablation_junction.sh > src/logs/ablation.log 2>&1 &
```

Five 2000-epoch operator runs, ~4 h each. One at a time, if you prefer:

```bash
export PYTHONPATH=$PWD/src
python src/dot_decoder/dot_deeponet.py --mode train --name abl-dot-branch \
       --pretrained_encoder TBD-N1-D128 --branch_hidden 512 512 512 512
python src/dot_decoder/dot_deeponet.py --mode eval  --name abl-dot-branch
```

Checkpoints stay in `dot_decoder/checkpoints/` rather than the main `src/checkpoints/`,
so `lower_bound/run.py` — which discovers models by globbing — does not pick the ablation
up unless it is asked to (`--ckpt_dir src/checkpoints src/dot_decoder/checkpoints`).
