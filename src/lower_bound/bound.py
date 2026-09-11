import numpy as np

def eigenvalues(I_S, verbose=True, tag="S"):
    """Eigenvalues lambda_k of Gamma_hat for the snapshot set I_S, through an SVD.

    I_S : (M, d) — snapshots AS ROWS (I_m = I_S[m]).
    Returns {lam, M, d} with lam in decreasing order, lambda_k = sigma_k^2/M.
    """
    import torch
    Z = (I_S.detach().to(torch.float64).contiguous().numpy()
         if isinstance(I_S, torch.Tensor)
         else np.array(I_S, dtype=np.float64, copy=True))
    M, d = Z.shape
    Z -= Z.mean(axis=0)

    sigma = np.linalg.svd(Z, compute_uv=False)
    lam = (sigma ** 2) / M

    if verbose:
        tr = float((Z ** 2).sum() / M)
        print(f"[bound:{tag}] M={M}  d={d}  Σλ={lam.sum():.6g}  tr(Γ̂)={tr:.6g}  "
              f"(gap {abs(lam.sum()-tr)/max(tr,1e-30):.1e})  "
              f"λ₁={lam[0]:.4g}  rank≈{int((lam > 1e-12*lam[0]).sum())}")
    return {"lam": lam, "M": M, "d": d}


def tail(spec, p):
    """sum_{k>p} lambda_k -- the right-hand side of Theorem 3.4."""
    lam = spec["lam"]
    return float(lam[p:].sum()) if p < lam.size else 0.0


def lower_bound(spec, p):
    """Lower bound on E_hat: sqrt(sum_{k>p} lambda_k)."""
    return float(np.sqrt(tail(spec, p)))


def energy_curve(spec, p_max=None):
    """The bound sqrt(sum_{k>p} lambda_k) for every p = 0..min(M, p_max), as (p, b)."""
    lam = np.asarray(spec["lam"], dtype=np.float64)
    tails = np.append(np.cumsum(lam[::-1])[::-1], 0.0)
    tails = np.maximum(tails, 0.0)
    p = np.arange(tails.size)
    if p_max is not None:
        p, tails = p[:p_max + 1], tails[:p_max + 1]
    return p, np.sqrt(tails)
