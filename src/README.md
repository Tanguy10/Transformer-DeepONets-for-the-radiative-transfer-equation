# Design notes

Why the code looks the way it does. See the top-level `README.md` for install, training
and evaluation.

## One angular coordinate, three columns

The 2D RTE has a single angular degree of freedom, and the code carries it as one number
everywhere: **φ**, the angle of the propagation direction measured from +y clockwise.
Every query point is a 3-column `(x, y, φ)` row, from the dataset down to the trunk (or
the coordinate decoder). `μ = (sin φ, cos φ)` is rebuilt in exactly one place,
`utils.dirs_from_phi` / `utils.transport_coeffs`, where the physics needs the vector:

```
  μ·∇I = sin φ · ∂I/∂x + cos φ · ∂I/∂y
```

φ_m = 2πm/16 are the FVDOM ordinates of the solver, so the 16 data directions land on
those angles exactly.

In front of the random Fourier features of the Transformer-Based DeepONet decoder, φ is
divided by 2π so it stays in [0, 1] like x and y — at σ = 3 the features would otherwise
see about six times more angular than spatial variation.

**The trade-off is real and deliberate.** Unlike a `(μx, μy)` unit-circle embedding, φ
does not make I structurally 2π-periodic: the network has to learn the periodicity, and
no training data straddles the φ = 0 ≡ 2π cut — which separates m = 15 from m = 0, two
directions that are neighbours on the circle.

## rel-L2 as the data loss

The default data loss is the per-snapshot relative L2 over the query points of the step,
averaged over the batch:

```
  L_data = mean_b  ||I_pred[b] - I_true[b]||₂ / ||I_true[b]||₂
```

This is also the metric that gets reported, and the criterion used to select the best
checkpoint — so what is optimised, what is monitored, and what is published are the same
quantity. `--data_loss mse` switches back to a pooled MSE divided by `I_scale²`.

## Two decoder families

| | how the inputs meet | decoder | file |
|---|---|---|---|
| DeepONet | **Hadamard product** of the branch nets (MIONet) | `c(u)·t(y)`, separable | `deeponet.py` |
| Transformer-Based DeepONet | one joint 2-channel FunDiff encoder | cross-attention on coordinates, **not separable** | `tbdeeponet.py` |

The Hadamard product encodes a multiplicative coupling between T and κ, consistent with
the RTE source S = κ·(σ/π)·T⁴. The Transformer-Based DeepONet decoder gives up
separability — and that is what buys it its accuracy, at the price of a residual that
needs a full double backward instead of the `jacfwd`/`vmap` shortcut a separable trunk
allows.

`deeponet_joint.py` also holds a third, intermediate variant (a joint convolutional
autoencoder feeding a single-branch DeepONet, `--mode train_deeponet`). It is kept for
reference and is not part of the reported tables.

## One softplus apart

`MultiBranchDeepONet` predicts `I = softplus(⟨β, γ(y)⟩ + b) · I_scale`. That softplus
enforces the physical positivity of the intensity at no parameter cost — it is what
Lu *et al.* do — but it is a nonlinear map of β, so the reconstructor is no longer
affine and Theorem 3.4 does not formally cover it.

`--affine` removes it. `I = (⟨β, γ(y)⟩ + b) · I_scale` is then *literally* the
`R(β) = γ₀ + Σ βₗ γₗ` of the theorem, and the bound of `lower_bound/` constrains the
model to the letter. The price is that I is free to go negative. Empirically the two
variants sit within a few tenths of a point of each other.

Three consequences, all of them deliberate:

* the choice is recorded in the state dict as an `affine_head` buffer, so
  `deeponet.load_model` restores the head from the checkpoint alone — the same idiom as
  `trunk_act_last`, and the reason `evaluate.py` never needs to be told;
* an affine run is named `<run>-affine` and **shares the encoder of `<run>`**
  (`encoders.encoder_run`): the output head is the only difference between the two, and
  a 90 MB encoder is not stored twice;
* `dot_decoder/` carries the same flag on its own decoder, for the same reason.

## Checkpoint contracts

* A DeepONet `.pth` is enough to rebuild its architecture AND its output head
  (`deeponet.infer_architecture`, `affine_head`), so `evaluate.py` never needs
  `--p_basis` and never needs `--affine`.
* A Transformer-Based DeepONet checkpoint holds the encoder, `dec_I`, the input
  statistics, `I_scale` and the coordinate normalisation together (`tbdeeponet.save_operator` /
  `load_operator`) — never re-pass the `--tbd_*` flags at evaluation time, they would be
  ignored anyway.
* An encoder fine-tuned during operator training is saved separately as
  `encoder_<kind>_<name>_ft.pt`, and evaluation prefers it: pairing a fine-tuned model
  with its original frozen encoder would mix inconsistent weights.

## Things that will bite

* **Snapshot index is not chronological.** `prepare_data.py` sorts the H5 groups
  alphabetically, so `time_10` comes before `time_9`. Recover the real time through
  `utils.snapshot_times`, never from the index.
* **The residual is monitored by default.** Even at `w_phys = 0` the RTE residual is
  measured on train and val and written to the history; measuring costs about four times
  less than optimising (59 ms against 236 ms per call for the Transformer-Based DeepONet).
  The curve cannot be recovered after the fact, hence the default.
* **`w_bc` is not wired** for the Transformer-Based DeepONet and raises if set: the domain
  boundary condition is not validated with certainty.
* **The solver H5 is probed only on request.** See `utils.solver_h5` — on the reference
  machine an unguarded `os.path.exists` on that path puts the process into
  uninterruptible disk sleep, where it stops answering SIGKILL.
