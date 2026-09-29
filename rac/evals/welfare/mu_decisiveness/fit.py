"""Thurstone Case V fit and bootstrap (upstream `fit.py`).

P(item_i > item_j) = Phi((mu_i - mu_j) / sqrt 2) with sigma fixed at 1, fitted
by maximum likelihood over the observed edges; mu is centred to fix the gauge.
torch is imported inside the fitting functions so `predict_matrix_caseV` stays
importable without it.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

_SQRT2 = math.sqrt(2.0)


def normalize_edges(rows: Sequence[dict]):
    """Edge rows -> (i_idx, j_idx, w_pos, w_neg). Sample-mode rows weight the
    likelihood by their win counts; logprob rows carry a probability and weight 1."""
    i_idx, j_idx, w_pos, w_neg = [], [], [], []
    for r in rows:
        i_idx.append(int(r["i"]))
        j_idx.append(int(r["j"]))
        if r.get("mode") == "sample":
            w_pos.append(float(r["wins_i"]))
            w_neg.append(float(r["wins_j"]))
        else:
            p = float(r["p_util"])
            w_pos.append(p)
            w_neg.append(1.0 - p)
    return (np.asarray(i_idx), np.asarray(j_idx),
            np.asarray(w_pos, dtype=np.float64), np.asarray(w_neg, dtype=np.float64))


def predict_matrix_caseV(mu) -> np.ndarray:
    """P_ij = Phi((mu_i - mu_j) / sqrt 2), diagonal set to 0.5."""
    import torch

    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float64)
    diff = mu_t[:, None] - mu_t[None, :]
    P = _phi_t(torch, diff / _SQRT2)
    P.fill_diagonal_(0.5)
    return P.numpy()


def fit_caseV_mle(rows: Sequence[dict], n: int, steps: int = 2000, lr: float = 0.05,
                  seed: int = 0, device: Optional[str] = None, mu_init=None) -> dict:
    """MLE for mu with sigma fixed at 1. `mu_init` warm-starts the optimiser."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if not len(rows):
        raise ValueError("fit_caseV_mle: no edges to fit")
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    ii = torch.as_tensor(i_idx, device=device)
    jj = torch.as_tensor(j_idx, device=device)
    wp = torch.as_tensor(w_pos, device=device)
    wn = torch.as_tensor(w_neg, device=device)
    if mu_init is None:
        mu = torch.zeros(n, dtype=torch.float64, device=device)
    else:
        mu = torch.as_tensor(np.asarray(mu_init), dtype=torch.float64, device=device).clone()
    mu.requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P = _phi_t(torch, (mu[ii] - mu[jj]) / _SQRT2).clamp(1e-9, 1 - 1e-9)
        nll = -(wp * torch.log(P) + wn * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu_c = mu - mu.mean()
    return {"mu": mu_c.detach().cpu().numpy()}


def _phi_t(torch, x):
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _batched_caseV_fit(torch, ii_be, jj_be, wp_be, wn_be, n, steps, lr, device,
                       mu_init=None) -> np.ndarray:
    """Fit B replicates simultaneously as one (B, n) mu tensor."""
    B = wp_be.shape[0]
    if mu_init is None:
        mu = torch.zeros(B, n, dtype=torch.float64, device=device)
    else:
        mi = torch.as_tensor(np.asarray(mu_init), dtype=torch.float64, device=device)
        mu = (mi.unsqueeze(0).expand(B, n).clone() if mi.ndim == 1 else mi.clone())
    mu.requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=lr)
    bidx = torch.arange(B, device=device).unsqueeze(1)
    for _ in range(steps):
        opt.zero_grad()
        diff = (mu[bidx, ii_be] - mu[bidx, jj_be]) / _SQRT2
        P = _phi_t(torch, diff).clamp(1e-9, 1 - 1e-9)
        nll = -(wp_be * torch.log(P) + wn_be * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        return (mu - mu.mean(dim=1, keepdim=True)).detach().cpu().numpy()


def bootstrap_measurement(rows: Sequence[dict], n: int, B: int, metric_fn, seed: int = 0,
                          steps: int = 600, lr: float = 0.05, device: Optional[str] = None,
                          mu_init=None) -> np.ndarray:
    """Resample edges with replacement (re-drawing sample-mode counts) and fit
    all B replicates in one batched optimisation."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    is_sample = np.array([r.get("mode") == "sample" for r in rows])
    totals = w_pos + w_neg
    E = len(rows)

    pick = rng.integers(0, E, size=(B, E))
    wp = w_pos[pick].copy()
    wn = w_neg[pick].copy()
    smask = is_sample[pick]
    if smask.any():
        Ns = totals[pick]
        ps = np.where(Ns > 0, wp / np.maximum(Ns, 1e-9), 0.5)
        draws = rng.binomial(np.where(smask, Ns, 0).astype(int), np.clip(ps, 0, 1))
        wp = np.where(smask, draws, wp)
        wn = np.where(smask, Ns - draws, wn)

    ii_be = torch.as_tensor(i_idx[pick], device=device)
    jj_be = torch.as_tensor(j_idx[pick], device=device)
    wp_be = torch.as_tensor(wp, device=device, dtype=torch.float64)
    wn_be = torch.as_tensor(wn, device=device, dtype=torch.float64)
    mu_b = _batched_caseV_fit(torch, ii_be, jj_be, wp_be, wn_be, n, steps, lr, device, mu_init)
    return np.array([metric_fn(mu_b[b]) for b in range(B)])


def bootstrap_items(rows: Sequence[dict], n: int, B: int, metric_fn, seed: int = 0,
                    steps: int = 400, lr: float = 0.05, device: Optional[str] = None,
                    mu_init=None) -> np.ndarray:
    """Cluster bootstrap over items. Drawing n items with replacement weights
    edge (i, j) by count[i] * count[j], so each edge's likelihood is weighted by
    that product instead of materialising the multigraph per replicate."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    i_idx, j_idx, w_pos, w_neg = normalize_edges(rows)
    E = len(rows)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    w_items = counts[:, i_idx] * counts[:, j_idx]
    ii_be = torch.as_tensor(i_idx, device=device).unsqueeze(0).expand(B, E)
    jj_be = torch.as_tensor(j_idx, device=device).unsqueeze(0).expand(B, E)
    wi = torch.as_tensor(w_items, device=device, dtype=torch.float64)
    wp_be = torch.as_tensor(w_pos, device=device, dtype=torch.float64).unsqueeze(0) * wi
    wn_be = torch.as_tensor(w_neg, device=device, dtype=torch.float64).unsqueeze(0) * wi
    mu_b = _batched_caseV_fit(torch, ii_be, jj_be, wp_be, wn_be, n, steps, lr, device, mu_init)
    return np.array([metric_fn(mu_b[b]) for b in range(B)])


def load_mu_init(prev_mu, items: List[str]) -> Optional[np.ndarray]:
    """Align a prior `{item: mu}` mapping (or a mu.json path) to `items` order."""
    import json
    from pathlib import Path

    if prev_mu is None:
        return None
    if isinstance(prev_mu, (str, Path)):
        prev_mu = json.loads(Path(prev_mu).read_text())
    arr = np.zeros(len(items), dtype=np.float64)
    overlap = 0
    for k, item in enumerate(items):
        if item in prev_mu:
            arr[k] = float(prev_mu[item])
            overlap += 1
    if items and overlap / len(items) < 0.5:
        raise ValueError(f"mu_init overlaps only {overlap}/{len(items)} items; "
                         "check that the item pool matches")
    return arr
