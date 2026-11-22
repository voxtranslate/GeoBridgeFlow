"""Preview pixel decoder: image latent (the anchored patch grid pooled to N tokens) -> a small RGB image.

It exists so that the image-writing direction produces something that can be looked at.  It is deliberately
tiny and trained with a contrast-weighted L1 loss on thumbnails (see ``GeoBridgeLoss.phase_a``), so its output
is a low-resolution preview, not a photograph; a pretrained unCLIP-style decoder on the same vision space is
the route to photorealism.  Its input is frozen (the anchored patch grid), so it shares no gradient with the
rest of the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import grid_factor
from .layers import Block


class PreviewDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, C, self.size = cfg.latent_dim, cfg.pix_width, cfg.thumb_size
        self.grid = grid_factor(cfg.n_tokens)
        self.inp = nn.Linear(d, C)
        self.pos = nn.Parameter(torch.randn(cfg.n_tokens, C) * 0.02)
        self.mix = Block(C, 4)
        chans = [C, C // 2, C // 4, C // 4]
        self.ups = nn.ModuleList([nn.Conv2d(chans[i], chans[i + 1], 3, padding=1) for i in range(3)])
        self.out = nn.Conv2d(chans[-1], 3, 3, padding=1)

    def forward(self, z_img: torch.Tensor) -> torch.Tensor:
        """z_img [B, N, d] -> RGB [B, 3, S, S] in [0, 1]."""
        h = self.mix(self.inp(z_img) + self.pos)
        h = h.transpose(1, 2).reshape(z_img.shape[0], -1, *self.grid)
        h = F.interpolate(h, size=(self.size // 8, self.size // 8), mode="bilinear", align_corners=False)
        for conv in self.ups:
            h = F.gelu(conv(F.interpolate(h, scale_factor=2, mode="nearest")))
        return torch.sigmoid(self.out(h))
