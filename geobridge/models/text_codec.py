"""Grapheme-Byte Compressor (GBC): tokenizer-free text <-> fixed-size continuous latent.

Encoder   UTF-8 bytes --(pooled per grapheme cluster)--> units --transformer--> resampler --> N x d latent
Decoder   byte-level autoregressive transformer cross-attending to the (noise-perturbed) latent

The encoder's sequence length is the number of *grapheme clusters*, so a Devanagari or Ethiopic sentence costs
the same number of positions as a Latin sentence with the same number of written symbols, and the latent has
the same size for every script.  There is no vocabulary to extend when a new script arrives.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..text_units import BOS, BYTE_VOCAB, EOS, PAD, bytes_to_text
from .layers import Block, Resampler, sincos, standardize_tokens


class TextCodec(nn.Module):
    def __init__(self, cfg, unit_vocab: int, unit_pad: int, unit_bytes: int, n_lang: int):
        super().__init__()
        self.cfg, W, d, N = cfg, cfg.text_width, cfg.latent_dim, cfg.n_tokens
        self.unit_pad = unit_pad
        self.unit_emb = nn.Embedding(unit_vocab, W, padding_idx=unit_pad)
        self.intra_pos = nn.Parameter(torch.zeros(unit_bytes, W))
        self.unit_proj = nn.Sequential(nn.Linear(2 * W, W), nn.GELU(), nn.Linear(W, W))
        self.enc = nn.ModuleList([Block(W, cfg.text_heads, dropout=cfg.dropout) for _ in range(cfg.text_enc_layers)])
        self.resampler = Resampler(W, N, d, layers=2, heads=cfg.text_heads, dropout=cfg.dropout)
        # decoder
        self.byte_emb = nn.Embedding(BYTE_VOCAB, W)
        self.lang_emb = nn.Embedding(n_lang, W)
        self.lat_in = nn.Linear(d, W)
        self.lat_pos = nn.Parameter(torch.randn(N, W) * 0.02)
        self.dec = nn.ModuleList([Block(W, cfg.text_heads, ctx_dim=W, dropout=cfg.dropout) for _ in range(cfg.text_dec_layers)])
        self.out_norm = nn.LayerNorm(W)
        self.out = nn.Linear(W, BYTE_VOCAB)

    # ------------------------------------------------------------------ encoder
    def embed_units(self, units: torch.Tensor) -> torch.Tensor:
        """units [B, L, M] -> unit embeddings [B, L, W] (mean + max pooling over the bytes of each unit)."""
        valid = (units != self.unit_pad).unsqueeze(-1)
        e = (self.unit_emb(units) + self.intra_pos[: units.shape[2]]) * valid
        mean = e.sum(2) / valid.sum(2).clamp_min(1)
        mx = e.masked_fill(~valid, -1e4).max(2).values
        mx = torch.where(valid.any(2), mx, torch.zeros_like(mx))
        return self.unit_proj(torch.cat([mean, mx], -1))

    def encode(self, units: torch.Tensor, unit_mask: torch.Tensor, return_units: bool = False):
        h0 = self.embed_units(units)
        h = h0 + sincos(torch.arange(h0.shape[1], device=h0.device), h0.shape[-1])
        for blk in self.enc:
            h = blk(h, mask=unit_mask)
        z = standardize_tokens(self.resampler(h, unit_mask))
        return (z, h0) if return_units else z

    def encode_from_embeddings(self, h0: torch.Tensor, unit_mask: torch.Tensor) -> torch.Tensor:
        """Same as :meth:`encode` but starting from unit embeddings (used for grapheme-level attribution)."""
        h = h0 + sincos(torch.arange(h0.shape[1], device=h0.device), h0.shape[-1])
        for blk in self.enc:
            h = blk(h, mask=unit_mask)
        return standardize_tokens(self.resampler(h, unit_mask))

    # ------------------------------------------------------------------ decoder
    def decoder(self) -> nn.Module:
        """The byte decoder as one module: a *view* on the same sub-modules and parameters (nothing is copied or
        renamed), so that the transport phase can train the decoder while the encoder stays frozen."""
        view = nn.Module()
        for name in ("byte_emb", "lang_emb", "lat_in", "dec", "out_norm", "out"):
            view.add_module(name, getattr(self, name))
        view.register_parameter("lat_pos", self.lat_pos)
        return view

    def decode_loss(self, z: torch.Tensor, tokens: torch.Tensor, lang: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced byte cross-entropy of ``tokens`` [B, T] (BOS ... EOS, PAD) given the latent z, and the
        byte accuracy."""
        logits = self._decode_states(z, tokens[:, :-1], lang)
        tgt = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), ignore_index=PAD)
        with torch.no_grad():
            m = tgt != PAD
            acc = ((logits.argmax(-1) == tgt) & m).sum().float() / m.sum().clamp_min(1)
        return loss, acc

    def _decode_states(self, z: torch.Tensor, tokens: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        ctx = self.lat_in(z) + self.lat_pos
        x = self.byte_emb(tokens) + sincos(torch.arange(tokens.shape[1], device=z.device), ctx.shape[-1])
        x = x + self.lang_emb(lang).unsqueeze(1)
        for blk in self.dec:
            x = blk(x, ctx=ctx, causal=True)
        return self.out(self.out_norm(x))

    def loss(self, txt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z = self.encode(txt["units"], txt["unit_mask"])
        zin = z
        if self.training and self.cfg.latent_noise_max > 0:  # noise-robust latent: flows never hit z exactly
            sigma = torch.rand(z.shape[0], 1, 1, device=z.device) * self.cfg.latent_noise_max
            sigma = sigma * (torch.rand_like(sigma) < 0.7)
            zin = z + sigma * torch.randn_like(z)
        loss, acc = self.decode_loss(zin, txt["bytes"], txt["lang"])
        return {"loss": loss, "z": z, "byte_acc": acc}

    @torch.no_grad()
    def generate(self, z: torch.Tensor, lang: torch.Tensor, max_len: Optional[int] = None) -> List[str]:
        max_len = max_len or self.cfg.max_text_bytes + 2
        B = z.shape[0]
        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=z.device)
        done = torch.zeros(B, dtype=torch.bool, device=z.device)
        for _ in range(max_len):
            nxt = self._decode_states(z, tokens, lang)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], 1)
            done |= nxt == EOS
            if done.all():
                break
        return [bytes_to_text(t[1:].tolist()) for t in tokens]
