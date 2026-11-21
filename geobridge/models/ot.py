"""Entropic optimal transport in the log domain, and the fused / semi-relaxed Gromov-Wasserstein solver used
for docking.  Everything runs in float32 outside autocast and uses only matrix products and logsumexp - no
SVD, no inversion - so it is safe under mixed-precision training and with small batches.

Fused objective (alpha in [0, 1], optional pinning cost R):

    FGW_alpha(Pi) = (1 - alpha) * sum_{ijkl} (C1_ik - C2_jl)^2 Pi_ij Pi_kl  +  alpha * <M, Pi>  +  <R, Pi>

    alpha = 0  pure Gromov-Wasserstein: compares only *relations inside* each set (robust to a displaced or
               rotated island, blind to where it sits)
    alpha = 1  pure Wasserstein with a cross-set cost M (sharp once the two sets share coordinates)

Marginals
    balanced       Pi 1 = a,  Pi^T 1 = b           (log-domain Sinkhorn inner loop)
    semi-relaxed   Pi 1 = a,  column marginal free (closed-form row-softmax inner step)
The semi-relaxed form lets an island batch use only the part of the hub bank it actually corresponds to,
instead of being forced to cover content it does not contain.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def log_sinkhorn(cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float, iters: int = 50) -> torch.Tensor:
    """Balanced entropic OT plan for a cost matrix [n, m]; fully in the log domain."""
    cost = cost.float()
    la, lb = torch.log(a.float().clamp_min(1e-30)), torch.log(b.float().clamp_min(1e-30))
    f, g = torch.zeros_like(a, dtype=torch.float32), torch.zeros_like(b, dtype=torch.float32)
    k = -cost / eps
    for _ in range(iters):
        f = eps * (la - torch.logsumexp(k + g.unsqueeze(0) / eps, dim=1))
        g = eps * (lb - torch.logsumexp(k + f.unsqueeze(1) / eps, dim=0))
    return torch.exp(k + f.unsqueeze(1) / eps + g.unsqueeze(0) / eps)


def gw_cost(C1: torch.Tensor, C2: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
    """Square-loss GW 'tensor product' L(C1, C2) (x) Pi  ->  [n, m]  in O(n^2 m + n m^2)."""
    a, b = Pi.sum(1), Pi.sum(0)
    const = (C1 ** 2 @ a).unsqueeze(1) + (C2 ** 2 @ b).unsqueeze(0)
    return const - 2.0 * C1 @ Pi @ C2.t()


@torch.no_grad()
def solve_fgw(C1: torch.Tensor, C2: torch.Tensor, M: Optional[torch.Tensor] = None, alpha: float = 0.0,
              eps: float = 0.05, outer: int = 10, inner: int = 50, marginal: str = "semi",
              pin: Optional[torch.Tensor] = None, a: Optional[torch.Tensor] = None,
              b: Optional[torch.Tensor] = None, init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Entropic (fused) GW by mirror descent.  Returns the coupling Pi [n, m] (no gradient)."""
    with torch.autocast(device_type=C1.device.type, enabled=False):
        C1, C2 = C1.float(), C2.float()
        n, m = C1.shape[0], C2.shape[0]
        a = torch.full((n,), 1.0 / n, device=C1.device) if a is None else a.float()
        b = torch.full((m,), 1.0 / m, device=C1.device) if b is None else b.float()
        Pi = a.unsqueeze(1) * b.unsqueeze(0) if init is None else init.float()
        for _ in range(outer):
            cost = torch.zeros(n, m, device=C1.device)
            if alpha < 1.0:
                cost = cost + (1.0 - alpha) * gw_cost(C1, C2, Pi)
            if M is not None and alpha > 0.0:
                cost = cost + alpha * M.float()
            if pin is not None:
                cost = cost + pin.float()
            if marginal == "semi":
                Pi = a.unsqueeze(1) * torch.softmax(-cost / eps, dim=1)
            else:
                Pi = log_sinkhorn(cost, a, b, eps, inner)
        return Pi


def fgw_value(C1: torch.Tensor, C2: torch.Tensor, Pi: torch.Tensor, M: Optional[torch.Tensor], alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Objective at a *fixed* coupling.  Differentiable w.r.t. C1 / M (envelope theorem: at the optimum the
    gradient through Pi vanishes, so Pi is treated as a constant).  Returns (total, gw term, w term)."""
    with torch.autocast(device_type=C1.device.type, enabled=False):
        C1, C2, Pi = C1.float(), C2.float(), Pi.float()
        gw = (gw_cost(C1, C2, Pi) * Pi).sum()
        w = (M.float() * Pi).sum() if M is not None else torch.zeros((), device=C1.device)
        return (1.0 - alpha) * gw + alpha * w, gw, w


def pairwise_cos_dist(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    y = x if y is None else torch.nn.functional.normalize(y.float(), dim=-1)
    return (1.0 - x @ y.t()).clamp_min(0.0)


def scale_free(C: torch.Tensor) -> torch.Tensor:
    """Divide a cost / structure matrix by its mean, so that eps has the same meaning in every space."""
    return C / C.mean().clamp_min(1e-8)


def coupling_stats(Pi: torch.Tensor) -> Dict[str, float]:
    rows = Pi / Pi.sum(1, keepdim=True).clamp_min(1e-30)
    ent = -(rows * torch.log(rows.clamp_min(1e-30))).sum(1)
    return {"row_entropy": float(ent.mean() / math.log(Pi.shape[1])), "max_prob": float(rows.max(1).values.mean()),
            "used_cols": float((Pi.sum(0) > 0.5 / Pi.shape[1] / Pi.shape[0] * Pi.shape[0]).float().mean())}


def row_confidence(Pi: torch.Tensor) -> torch.Tensor:
    """1 - normalised row entropy: 1 = the island sample has one clear partner, 0 = it could be anything."""
    rows = Pi / Pi.sum(1, keepdim=True).clamp_min(1e-30)
    ent = -(rows * torch.log(rows.clamp_min(1e-30))).sum(1)
    return (1.0 - ent / math.log(Pi.shape[1])).clamp(0.0, 1.0)
