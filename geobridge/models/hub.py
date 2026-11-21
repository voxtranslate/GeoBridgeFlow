"""The hub: a Barycentric Visual Interlingua (BVI), plus the anchors it is ablated against.

An utterance (text or speech, any language) is turned into K_h *slot queries*.  Each slot attends over the
frozen patch tokens of the image the utterance describes; the hub token of that slot is the attention-weighted
**barycentre** of those frozen tokens:

        z*_k = sum_p  alpha_kp * v_p ,     alpha_k = softmax_p( <q_k, key(v_p)> / tau )

so a hub code is, by construction, a point of conv(V_image)^{K_h}: it lives in the frozen vision geometry,
contains no text, and is sentence-level (two captions of one image attend differently and get different
codes).  It can be drawn on the image, which makes the interlingua directly inspectable.

The grounder is trained with three terms.  (i) A late-interaction InfoNCE between utterances and images: which
patch does each slot find.  (ii) An **explanatory** term: a small predictor must reconstruct the utterance latent
from the hub code alone, so the slots have to cover *everything that was said*, in a consistent order.  The
contrastive term cannot do this by itself: it is satisfied as soon as one slot finds one matching patch, and
the hub then describes a single object per sentence (measured: 0 % of two-object captions had both objects
covered).  (iii) A same-language text<->speech agreement term.  No loss ever pairs two languages.

Anchors available for the ablation (``cfg.hub_mode``):
    bary    (ours) utterance-conditioned barycentres of frozen patch tokens
    global  scene-level visual pivot: a fixed pooling of the image, identical for all its captions
    free    unconstrained learned anchor in the spirit of FlowBind: hub = f(modality latent), no vision geometry
    text    text anchor: the first K_h tokens of a hub-language text latent (image-pivoted for the island)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import grid_factor
from .layers import Block


class VisualInterlingua(nn.Module):
    MODS = {"txt": 0, "aud": 1}

    def __init__(self, cfg, vision_dim: int, n_vis_tokens: int, anchor: Dict[str, torch.Tensor], n_lang: int):
        super().__init__()
        self.cfg, W, d = cfg, cfg.hub_width, cfg.latent_dim
        self.K, self.P = cfg.hub_slots, n_vis_tokens  # P counts the global token (last)
        self.register_buffer("a_mean", anchor["mean"].clone())
        self.register_buffer("a_basis", anchor["basis"].clone())
        self.register_buffer("a_scale", anchor["scale"].clone())
        self.lat_in = nn.Linear(d, W)
        self.lat_pos = nn.Parameter(torch.randn(cfg.n_tokens, W) * 0.02)
        self.mod_emb = nn.Embedding(2, W)
        self.lang_emb = nn.Embedding(n_lang, W)
        self.slots = nn.Parameter(torch.randn(self.K, W))  # unit scale: slots must start out *different*
        self.blocks = nn.ModuleList([Block(W, cfg.hub_heads, ctx_dim=W) for _ in range(cfg.hub_layers)])
        self.norm = nn.LayerNorm(W)
        self.q_out = nn.Linear(W, W)
        self.key = nn.Linear(d, W)
        self.free_out = nn.Linear(W, d)          # used by hub_mode == "free"
        self.explainer = Explainer(cfg, n_lang)
        side = int(round((self.P - 1) ** 0.5))
        self.grid = (side, side) if side * side == self.P - 1 else (1, self.P - 1)

    # ------------------------------------------------------------------ frozen anchor geometry
    def project(self, img: torch.Tensor) -> torch.Tensor:
        """Raw vision tokens [B, P, D_v] -> anchored tokens [B, P, d] (frozen PCA isometry + one global scale)."""
        return ((img.float() - self.a_mean) @ self.a_basis) * self.a_scale

    def image_latent(self, V: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Image-side modality latent: the anchored patch grid pooled to n_tokens tokens."""
        B, _, d = V.shape
        r, c = grid_factor(n_tokens)
        g = V[:, :-1].transpose(1, 2).reshape(B, d, *self.grid)
        return F.adaptive_avg_pool2d(g, (r, c)).flatten(2).transpose(1, 2)

    def scene_code(self, V: torch.Tensor) -> torch.Tensor:
        """Scene-level pivot (hub_mode == "global"): global token + pooled regions, same for every caption."""
        if self.K == 1:
            return V[:, -1:]
        return torch.cat([V[:, -1:], self.image_latent(V, self.K - 1)], 1)

    # ------------------------------------------------------------------ utterance -> slot queries
    def slot_features(self, z: torch.Tensor, mod: str, lang: torch.Tensor) -> torch.Tensor:
        ctx = self.lat_in(z) + self.lat_pos + self.mod_emb.weight[self.MODS[mod]] + self.lang_emb(lang).unsqueeze(1)
        h = self.slots.unsqueeze(0).expand(z.shape[0], -1, -1)
        for blk in self.blocks:
            h = blk(h, ctx=ctx)
        return self.norm(h)

    def queries(self, z: torch.Tensor, mod: str, lang: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_out(self.slot_features(z, mod, lang)), dim=-1)

    def keys(self, V: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key(V), dim=-1)

    def attend(self, q: torch.Tensor, V: torch.Tensor, keys: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """q [n, K, W], V [n, P, d] (the utterance's own image) -> hub [n, K, d], attention [n, K, P-1].

        Slots attend over the *patch* tokens only.  The global token is excluded on purpose: it is the mean
        of the image, and a slot that settles on it turns the hub back into a scene-level pivot (every
        caption of an image gets the same code).  This collapse does happen if the token is offered."""
        keys = self.keys(V) if keys is None else keys
        attn = torch.softmax(torch.einsum("nkw,npw->nkp", q, keys[:, :-1]).float() / self.cfg.hub_temp, dim=-1)
        return torch.einsum("nkp,npd->nkd", attn.to(V.dtype), V[:, :-1]), attn

    def similarity(self, q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Late-interaction score between every utterance and every image: mean over slots of the best patch.
        q [n, K, W], keys [B, P, W] -> [n, B]"""
        return torch.einsum("nkw,bpw->nbkp", q, keys[:, :-1]).amax(-1).mean(-1)

    def free_code(self, z: torch.Tensor, mod: str, lang: torch.Tensor) -> torch.Tensor:
        return self.free_out(self.slot_features(z, mod, lang))

    # ------------------------------------------------------------------ retrieval in hub space
    @staticmethod
    def hub_to_image_scores(hub: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """How well each hub code [n, K, d] is explained by the patches of each gallery image [G, P, d].
        Grounder-free: used to score hub estimates that came out of a backward ODE.  -> [n, G]"""
        h, v = F.normalize(hub.float(), dim=-1), F.normalize(V[:, :-1].float(), dim=-1)
        return torch.einsum("nkd,gpd->ngkp", h, v).amax(-1).mean(-1)


class Explainer(nn.Module):
    """hub code [n, K_h, d] -> predicted modality latent [n, N, d].  Only used in phase A, to make the grounder
    select the patches that explain the utterance (the t = 0 view of flow matching: what the hub does not
    determine is unexplained variance of the latent).  Discarded afterwards; the flows take over."""

    def __init__(self, cfg, n_lang: int):
        super().__init__()
        W, d = cfg.hub_width, cfg.latent_dim
        self.inp = nn.Linear(d, W)
        self.slot_pos = nn.Parameter(torch.randn(cfg.hub_slots, W) * 0.02)
        self.mod_emb = nn.Embedding(2, W)
        self.lang_emb = nn.Embedding(n_lang, W)
        self.queries = nn.Parameter(torch.randn(cfg.n_tokens, W) * 0.02)
        self.blocks = nn.ModuleList([Block(W, cfg.hub_heads, ctx_dim=W) for _ in range(2)])
        self.norm = nn.LayerNorm(W)
        self.out = nn.Linear(W, d)

    def forward(self, hub: torch.Tensor, mod: int, lang: torch.Tensor) -> torch.Tensor:
        ctx = self.inp(hub) + self.slot_pos + self.mod_emb.weight[mod] + self.lang_emb(lang).unsqueeze(1)
        q = self.queries.unsqueeze(0).expand(hub.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk(q, ctx=ctx)
        return self.out(self.norm(q))


class HubBank(nn.Module):
    """FIFO memory of hub codes produced from image-paired utterances: the empirical hub distribution that
    unpaired islands are docked onto.  Stored as buffers, so it is checkpointed with the model."""

    def __init__(self, size: int, k: int, d: int):
        super().__init__()
        self.register_buffer("codes", torch.zeros(size, k, d))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def push(self, z: torch.Tensor) -> None:
        z = z.detach().float()
        size = self.codes.shape[0]
        if z.shape[0] >= size:
            z = z[-size:]
        idx = (self.ptr + torch.arange(z.shape[0], device=z.device)) % size
        self.codes[idx] = z
        self.ptr.copy_((self.ptr + z.shape[0]) % size)
        self.count.copy_(torch.clamp(self.count + z.shape[0], max=size))

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        c = int(self.count)
        idx = torch.randperm(c, device=self.codes.device)[: min(n, c)]
        return self.codes[idx]

    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        c = max(1, int(self.count))
        return self.codes[:c].mean(0, keepdim=True)
