"""Hub (+) private bidirectional flow spokes.

One spoke per modality.  Its state is N = K_h + K_p tokens of width d:

    t = 0   x_0 = [ z* (K_h hub tokens) ; xi (K_p private tokens, xi ~ N(0, I)) ]
    t = 1   x_1 = z_m  (the modality latent)
    path    x_t = (1 - t) x_0 + t x_1 ,   target velocity  u = x_1 - x_0

Forward integration (0 -> 1) *generates* the modality from a hub code; the private tokens absorb everything
the hub does not determine (wording, speaker, phrasing), so one hub code maps to many valid outputs.
Backward integration (1 -> 0) *reads* a modality sample and returns both its hub code (first K_h tokens) and
its private code.  The velocity field is a shared transformer per modality whose linear maps carry a
rank-r LoRA pair per language.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .layers import Block, sincos


class VelocityNet(nn.Module):
    def __init__(self, cfg, n_lang: int, n_private: Optional[int] = None):
        super().__init__()
        n_private = cfg.private_slots if n_private is None else n_private
        W, d, N = cfg.flow_width, cfg.latent_dim, cfg.hub_slots + n_private
        self.W = W
        self.inp = nn.Linear(d, W)
        self.pos = nn.Parameter(torch.randn(N, W) * 0.02)
        self.kind = nn.Parameter(torch.randn(2, W) * 0.02)  # hub slot vs private slot
        self.register_buffer("kind_idx", torch.cat([torch.zeros(cfg.hub_slots), torch.ones(n_private)]).long())
        self.time = nn.Sequential(nn.Linear(W, W), nn.SiLU(), nn.Linear(W, W))
        self.lang_emb = nn.Embedding(max(1, n_lang), W)
        rank = cfg.lora_rank if n_lang > 0 else 0
        self.blocks = nn.ModuleList([Block(W, cfg.flow_heads, n_lang=n_lang, rank=rank, alpha=cfg.lora_alpha, cond_dim=W)
                                     for _ in range(cfg.flow_layers)])
        self.out_norm = nn.LayerNorm(W)
        self.out = nn.Linear(W, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        cond = self.time(sincos(t * 1000.0, self.W)) + self.lang_emb(lang)
        h = self.inp(x) + self.pos + self.kind[self.kind_idx]
        for blk in self.blocks:
            h = blk(h, lang=lang, cond=cond)
        return self.out(self.out_norm(h))


class Spoke(nn.Module):
    def __init__(self, cfg, n_lang: int, n_private: Optional[int] = None):
        """``n_private``: private tokens of this spoke's latent (default cfg.private_slots)."""
        super().__init__()
        self.cfg, self.K = cfg, cfg.hub_slots
        self.n_private = cfg.private_slots if n_private is None else n_private
        self.vel = VelocityNet(cfg, n_lang, self.n_private)

    # ------------------------------------------------------------------ source construction
    def source(self, hub: torch.Tensor, temp: float = 1.0, private: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[hub ; private].  ``private`` given -> reuse it; otherwise draw N(0, temp^2) (zeros if disabled)."""
        B, _, d = hub.shape
        if private is None:
            private = torch.randn(B, self.n_private, d, device=hub.device, dtype=hub.dtype) * temp
            if not self.cfg.private_noise:
                private = torch.zeros_like(private)
        return torch.cat([hub, private], 1)

    # ------------------------------------------------------------------ conditional flow matching
    def cfm_loss(self, x0: torch.Tensor, x1: torch.Tensor, lang: torch.Tensor,
                 weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x0.shape[0]
        if self.cfg.t_sampling == "logit_normal":
            t = torch.sigmoid(torch.randn(B, device=x0.device))
        else:
            t = torch.rand(B, device=x0.device)
        x1 = x1[:, : x0.shape[1]]   # (a speech latent can carry listening tokens after the ones that are written)
        tt = t.view(B, 1, 1).to(x0.dtype)
        v = self.vel((1 - tt) * x0 + tt * x1, t, lang)
        per = ((v.float() - (x1 - x0).float()) ** 2).mean(dim=(1, 2))
        if weight is not None:
            return (per * weight).sum() / weight.sum().clamp_min(1e-6)
        return per.mean()

    # ------------------------------------------------------------------ ODE integration
    def solve(self, x: torch.Tensor, lang: torch.Tensor, direction: int, steps: Optional[int] = None,
              solver: Optional[str] = None, grad: bool = False, return_path: bool = False,
              t_start: Optional[float] = None):
        """direction = +1: t t_start -> 1 (hub -> modality; t_start defaults to 0);
        direction = -1: t t_start -> 0 (modality -> hub; t_start defaults to 1)."""
        steps, solver = steps or self.cfg.ode_steps, solver or self.cfg.ode_solver
        x = x[:, : self.K + self.n_private]   # the written part of a latent
        t0 = (0.0 if direction > 0 else 1.0) if t_start is None else float(t_start)
        h = direction * ((1.0 - t0) if direction > 0 else t0) / steps
        t = torch.full((x.shape[0],), t0, device=x.device)
        path: List[torch.Tensor] = [x]
        with torch.set_grad_enabled(grad):
            for _ in range(steps):
                if solver == "euler":
                    x = x + h * self.vel(x, t, lang)
                elif solver == "heun":
                    k1 = self.vel(x, t, lang)
                    k2 = self.vel(x + h * k1, t + h, lang)
                    x = x + h * 0.5 * (k1 + k2)
                else:  # midpoint
                    k1 = self.vel(x, t, lang)
                    x = x + h * self.vel(x + 0.5 * h * k1, t + 0.5 * h, lang)
                t = t + h
                if return_path:
                    path.append(x)
        return (x, path) if return_path else x

    def read(self, z: torch.Tensor, lang: torch.Tensor, grad: bool = False, steps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """modality latent -> (hub estimate [B, K_h, d], private code [B, K_p, d]) by the backward ODE."""
        x0 = self.solve(z, lang, -1, steps=steps, grad=grad)
        return x0[:, : self.K], x0[:, self.K:]

    def write(self, hub: torch.Tensor, lang: torch.Tensor, temp: Optional[float] = None,
              private: Optional[torch.Tensor] = None, grad: bool = False, steps: Optional[int] = None,
              draft: Optional[torch.Tensor] = None, t0: float = 0.0) -> torch.Tensor:
        """hub code -> modality latent.

        ``draft`` (the Explainer's one-pass prediction of the latent) with ``t0`` in (0, 1] starts the ODE on
        the straight path between the source and the draft, at time t0, instead of at the source: the flow
        only has to finish what the draft began (t0 = 1 returns the draft itself, with no ODE step)."""
        temp = self.cfg.sample_temp if temp is None else temp
        x = self.source(hub, temp, private)
        if draft is not None and draft.shape[1] < x.shape[1]:   # tokens the draft does not cover start from the source
            draft = torch.cat([draft, x[:, draft.shape[1]:]], 1)
        if draft is None or t0 <= 0.0:
            return self.solve(x, lang, +1, steps=steps, grad=grad)
        if t0 >= 1.0:
            return draft
        steps = max(1, int(round((steps or self.cfg.ode_steps) * (1.0 - t0))))
        return self.solve((1.0 - t0) * x + t0 * draft, lang, +1, steps=steps, grad=grad, t_start=t0)
