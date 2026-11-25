"""Building blocks: language-factorised LoRA linear, attention, pre-LN transformer block, Perceiver resampler."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos(pos: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of arbitrary (possibly fractional) positions.  pos [...] -> [..., dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=pos.device, dtype=torch.float32) / half)
    args = pos.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B_l A_l x, with one low-rank pair (A_l, B_l) per language l.

    The dense weight is the shared, language-agnostic drift; the low-rank pair is everything a language owns.
    B is zero-initialised, so a new language starts exactly at the shared field.
    """

    def __init__(self, in_f: int, out_f: int, n_lang: int = 0, rank: int = 0, alpha: float = 16.0, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=bias)
        self.rank = rank if n_lang > 0 else 0
        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.randn(n_lang, rank, in_f) / math.sqrt(in_f))
            self.lora_B = nn.Parameter(torch.zeros(n_lang, out_f, rank))
            self.scale = alpha / rank

    def forward(self, x: torch.Tensor, lang: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.lin(x)
        if self.rank > 0 and lang is not None:
            a, b = self.lora_A[lang].to(x.dtype), self.lora_B[lang].to(x.dtype)  # [B, r, in], [B, out, r]
            y = y + self.scale * torch.einsum("bnr,bor->bno", torch.einsum("bni,bri->bnr", x, a), b)
        return y


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, ctx_dim: Optional[int] = None, n_lang: int = 0, rank: int = 0,
                 alpha: float = 16.0):
        super().__init__()
        self.heads, self.cross = heads, ctx_dim is not None
        self.q = LoRALinear(dim, dim, n_lang, rank, alpha)
        self.kv = LoRALinear(ctx_dim or dim, 2 * dim, n_lang, rank, alpha)
        self.proj = LoRALinear(dim, dim, n_lang, rank, alpha)

    def forward(self, x, ctx=None, key_mask=None, causal: bool = False, lang=None):
        B, T, D = x.shape
        src = ctx if ctx is not None else x
        S, H = src.shape[1], self.heads
        q = self.q(x, lang).view(B, T, H, D // H).transpose(1, 2)
        k, v = self.kv(src, lang).view(B, S, 2, H, D // H).permute(2, 0, 3, 1, 4)
        mask = None
        if key_mask is not None:
            mask = key_mask[:, None, None, :].expand(B, 1, T, S)
        if causal:
            tri = torch.ones(T, S, dtype=torch.bool, device=x.device).tril()[None, None]
            mask = tri if mask is None else (mask & tri)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, T, D), lang)


class Block(nn.Module):
    """Pre-LN transformer block: self-attention [+ cross-attention] + MLP.

    ``cond_dim > 0`` turns the norms into AdaLN (scale/shift predicted from a conditioning vector,
    zero-initialised so the block starts as a plain pre-LN block).  ``n_lang, rank`` enable language LoRA.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, ctx_dim: Optional[int] = None, n_lang: int = 0,
                 rank: int = 0, alpha: float = 16.0, cond_dim: int = 0, self_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.has_self, self.has_cross, self.cond = self_attn, ctx_dim is not None, cond_dim > 0
        self.drop = nn.Dropout(dropout)   # on every residual branch (identity when dropout = 0)
        affine = not self.cond
        if self_attn:
            self.n1 = nn.LayerNorm(dim, elementwise_affine=affine)
            self.attn = Attention(dim, heads, None, n_lang, rank, alpha)
        if self.has_cross:
            self.nc = nn.LayerNorm(dim, elementwise_affine=affine)
            self.xattn = Attention(dim, heads, ctx_dim, n_lang, rank, alpha)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=affine)
        hid = int(dim * mlp_ratio)
        self.fc1 = LoRALinear(dim, hid, n_lang, rank, alpha)
        self.fc2 = LoRALinear(hid, dim, n_lang, rank, alpha)
        if self.cond:
            n_norm = int(self_attn) + int(self.has_cross) + 1
            self.ada = nn.Linear(cond_dim, 2 * dim * n_norm)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)

    def forward(self, x, mask=None, ctx=None, ctx_mask=None, causal: bool = False, lang=None, cond=None):
        mods = None
        if self.cond:
            mods = list(self.ada(cond).unsqueeze(1).chunk(2 * (int(self.has_self) + int(self.has_cross) + 1), dim=-1))

        def norm(layer, h):
            h = layer(h)
            if mods is not None:
                s, b = mods.pop(0), mods.pop(0)
                h = h * (1 + s) + b
            return h

        if self.has_self:
            x = x + self.drop(self.attn(norm(self.n1, x), key_mask=mask, causal=causal, lang=lang))
        if self.has_cross:
            x = x + self.drop(self.xattn(norm(self.nc, x), ctx=ctx, key_mask=ctx_mask, lang=lang))
        return x + self.drop(self.fc2(F.gelu(self.fc1(norm(self.n2, x), lang)), lang))


class Resampler(nn.Module):
    """Perceiver-style resampler: n learned queries read a variable-length sequence -> fixed [n, out_dim]."""

    def __init__(self, dim: int, n_queries: int, out_dim: int, layers: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, ctx_dim=dim, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``anchor`` [B, n, dim]: what every query starts from in addition to its learned vector (e.g. the mean of
        its own span of the sequence, which makes the n outputs an ordered summary instead of a free set)."""
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        if anchor is not None:
            q = q + anchor
        for blk in self.blocks:
            q = blk(q, ctx=x, ctx_mask=mask)
        return self.out(self.norm(q))


def standardize_tokens(z: torch.Tensor) -> torch.Tensor:
    """Token-wise standardisation (parameter-free LayerNorm).  Every latent token has zero mean and unit
    variance across channels, so all modality latents live at the same scale without tracked statistics."""
    return F.layer_norm(z, (z.shape[-1],))


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int, zero_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.GELU(), nn.Linear(d_hid, d_out))
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
