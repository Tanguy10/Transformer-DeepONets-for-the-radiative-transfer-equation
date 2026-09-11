# `lower_bound/` — Theorem 3.4 as a number attached to this dataset

**Figure 7 of the paper.** For each rank p, the smallest error an affine reconstructor
can reach on this dataset — once on the training set, once on the test set — with the
models of Table II placed on it.

```bash
setsid nohup bash scripts/lower_bound.sh > src/logs/lower_bound.log 2>&1 &

# what will be compared, and where it is read from
python src/lower_bound/run.py --list
# a subset, by shell pattern
python src/lower_bound/run.py --models 'PCA-*' 'TBD-N?-D128' --out tbd_vs_pca.pdf
# bring the junction ablation of dot_decoder/ into the same figure
python src/lower_bound/run.py --ckpt_dir src/checkpoints src/dot_decoder/checkpoints
```

Every model sitting in `--ckpt_dir` enters the comparison: its family, its width p and
whether its reconstructor is affine are all read off the checkpoint, never listed by
hand. `p` is `p_basis` for the DeepONets and `N·D` for the Transformer-Based DeepONet,
whose attention decoder reads the whole latent — the bound only *constrains* the affine
rows, and for the Transformer p merely places it at a comparable width.

The cost is one eigenvalue problem of the size of the dataset (M×M, M ≤ 4200) plus one
pass of each model over train and test; `--refresh` controls what is recomputed rather
than read from `cache/`.

⚠ Only the `-affine` runs are covered *to the letter*. A softplus head is a monotone map
on top of the affine junction and formally falls outside the theorem — see the
"One softplus apart" section of `src/README.md`.

---

## Goal

Establish a theoretical lower bound on DeepONets whose final junction is affine. If,
empirically, the Transformer-Based DeepONet (whose reconstructor is nonlinear) goes below
that bound, we have proof that a vanilla DeepONet is limited on this problem.

## 1) Architecture

*(Notation of "Error estimates for DeepONets: a deep learning framework in infinite
dimensions")*

Any DeepONet can be written as

$$\tilde{G} = R \circ A \circ E \qquad \text{with}$$

$$E : X \to \mathbb{R}^m \quad \text{the encoder}$$

$$A : \mathbb{R}^m \to \mathbb{R}^p \quad \text{the approximator}$$

$$R : \mathbb{R}^p \to Y \quad \text{the reconstructor}$$

In a vanilla DeepONet, $R$ is the branch/trunk dot product:

$$R(\beta) = \gamma_0 + \sum_{l=1}^{p} \beta_l \gamma_l$$

and is therefore affine in $\beta$ (linear plus the bias $\gamma_0$), where $\beta$ is the
output vector of the branch net and $\gamma$ that of the trunk net.

**Remark.** The Transformer-Based DeepONet / CViT models do not have this affine
structure: their reconstructor is nonlinear, being based on an attention mechanism.

## 2) The continuous setting of "Error estimates for DeepONets"

Let $\mu$ be a probability measure on $X$ (the space of input functions, i.e. where $T$
and $\kappa$ live). All the theory below holds for $\mu$ such that the measured error is

$$\hat{\mathcal{E}}^2 = \iint_{X,\,U} \Big| \underbrace{G(u)(y)}_{\text{true operator}} - \underbrace{\hat{G}(u)(y)}_{\text{model}} \Big|^2 \, dy \, d\mu(x)$$

(cf. page 9 of the paper)

We rely on Theorem 3.4 of the paper, which states that, whatever $E$ and $A$ are, if $R$
is affine (as it is in a vanilla DeepONet) then, setting

$$\Gamma_{G_{\#}\mu} = \int_Y \big(v - \mathbb{E}_\nu(v)\big) \otimes \big(v - \mathbb{E}_\nu(v)\big) \, d\nu(v)$$

where $\nu = G_{\#}\mu$,

and $\lambda_1 \ge \dots \ge \lambda_p \ge \dots \ge 0$ the eigenvalues of that operator,
then

$$\boxed{\hat{\mathcal{E}}^2 \ge \sum_{k > p} \lambda_k}$$

The measures of interest here are those of the training and of the test datasets. Both
have the same form, so what follows is written identically for either.

## 3) Moving to a dataset

Take a database of $M$ snapshots $I_m$, associated with temperature and $\kappa$ field
vectors

$$(T_m, \kappa_m, I_m), \qquad I_m \in \mathbb{R}^d$$

Orders of magnitude:

$$d = 150 \times 150 \times 16 = 360\,000$$

$$M \approx 10^3$$

Define the empirical measures

$$\hat{\mu}_M = \frac{1}{M}\sum_{m=1}^{M} \delta_{(T_m, \kappa_m)} \qquad\qquad \hat{\nu}_M = \frac{1}{M}\sum_{m=1}^{M} \delta_{I_m}$$

so that (provided the boundary conditions are the same throughout the dataset)

$$\hat{\nu}_M = G_{\#}\hat{\mu}_M$$

**Note.** The errors of the solver that produced the database are also neglected.

We then define

$$\bar{I} = \frac{1}{M}\sum_{m=1}^{M} I_m$$

$$\hat{\Gamma}_M = \frac{1}{M}\sum_{m=1}^{M} (I_m - \bar{I}) \otimes (I_m - \bar{I}) \in \mathbb{R}^{d \times d}$$

Theorem 3.4 then gives a lower bound — provided the eigenvalues of $\hat{\Gamma}_M$ can
be computed. The problem: $d \gg 10^5$, which makes that hard.

## 4) The dimension trick

Define

$$U : \begin{cases} \mathbb{R}^M \to \mathbb{R}^d \\ c \mapsto \displaystyle\sum_{m=1}^{M} c_m (I_m - \bar{I}) \end{cases}$$

and $U^*$:

$$U^* : \begin{cases} \mathbb{R}^d \to \mathbb{R}^M \\ v \mapsto \big(\langle I_m - \bar{I},\, v \rangle_{\mathbb{R}^d}\big)_{m=1}^{M} \end{cases}$$

so that $U^*$ is the adjoint of $U$ for the canonical inner product on $\mathbb{R}^d$.

We then have

$$\hat{\Gamma} = \frac{1}{M} U U^*$$

$\hat{\Gamma}$ is factorised.

**Algebraic fact.** $UU^*$ and $U^*U$ have the same non-zero eigenvalues.

Here $M \ll d$, and $U^*U \in \mathbb{R}^{M \times M}$, so we look for the eigenvalues of
$U^*U$ instead.

## 5) Application

We apply this to the training and to the test datasets.

**Careful:** applied to train, for instance, the inequality bounds the *training* error:

$$\hat{\mathcal{E}}^2_{\text{train}} \ge \sum_{k > p} \lambda_{k,\text{train}}$$

and likewise

$$\hat{\mathcal{E}}^2_{\text{test}} \ge \sum_{k > p} \lambda_{k,\text{test}}$$

The test dataset therefore has to be non-negligible.

---

## References

- S. Lanthaler, S. Mishra, G. E. Karniadakis, *Error estimates for DeepONets: a deep learning framework in infinite dimensions*, Transactions of Mathematics and Its Applications, Vol. 6, Issue 1, 2022, tnac001. [https://doi.org/10.1093/imatrm/tnac001](https://doi.org/10.1093/imatrm/tnac001) (preprint: [arXiv:2102.09618](https://arxiv.org/abs/2102.09618))
