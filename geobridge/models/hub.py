"""The hub: a Barycentric Visual Interlingua (BVI), plus the anchors it is ablated against.

An utterance (text or speech, any language) is turned into K_h *slot queries*.  The slots attend, one after
the other, over the frozen patch tokens of the image the utterance describes; the hub token of a slot is the
attention-weighted **barycentre** of those frozen tokens:

        z*_k = sum_p  alpha_kp * v_p ,     alpha_k = softmax_p( <q_k, key(v_p)> / tau  -  inhibition_k )

(each slot is inhibited on what earlier slots took and may choose "null").  A hub code is, by construction,
a point of conv(V_image + origin)^{K_h}: it lives in the frozen vision geometry, contains no text, and is
sentence-level.  It can be drawn on the image, which makes the interlingua directly inspectable.

Mutual sufficiency.  The grounder is trained so that the hub code is the information the sentence and the
image have in common:

    L_ground  contrastive, under the same sequential attention - the code is found in the right image
    L_expl    Explainer(z*) must reconstruct the utterance latent - the code says everything the sentence says
    L_min     an image-free Reader(utterance) must predict z*     - the code says nothing the sentence does not
    L_agree   text and speech of one caption (same language) get the same code

L_min is what removes slots that look at objects the sentence never mentions: such a token cannot be
predicted from the sentence, a mentioned object or a null slot can.  The Reader is not thrown away: it is the
test-time read path (utterance -> hub in one pass, no image, no ODE), and the Explainer is the writer's draft.
No loss ever pairs two languages.

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


def hub_view(z: torch.Tensor, n_content: int, n_written: int) -> torch.Tensor:
    """[content | (realisation) | (listening)] -> [content | listening]: the tokens of a latent that the hub reads."""
    if z.shape[1] > n_written:
        return torch.cat([z[:, :n_content], z[:, n_written:]], 1)
    return z[:, :n_content]


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
        self.lat_pos = nn.Parameter(torch.randn(cfg.max_tokens, W) * 0.02)   # (latents can differ in length)
        self.visible = {m: cfg.hub_tokens_of(m) for m in self.MODS}   # tokens of each latent that the hub reads
        self.written = {m: cfg.tokens_of(m) for m in self.MODS}       # ... tokens a writer writes (listening tokens follow)
        self.mod_emb = nn.Embedding(2, W)
        self.lang_emb = nn.Embedding(n_lang, W)
        self.slots = nn.Parameter(torch.randn(self.K, W))  # unit scale: slots must start out *different*
        self.blocks = nn.ModuleList([Block(W, cfg.hub_heads, ctx_dim=W, dropout=cfg.dropout) for _ in range(cfg.hub_layers)])
        self.norm = nn.LayerNorm(W)
        self.q_out = nn.Linear(W, W)
        self.key = nn.Linear(d, W)
        self.null_key = nn.Parameter(torch.randn(W) * 0.1)
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.null_enabled = True   # the trainer switches it off for the first part of phase A (not a parameter)
        self.temp_scale = 1.0      # > 1 early in phase A: soft attention, annealed to the sharp final value
        self.explainer = Explainer(cfg, n_lang)   # hub -> utterance latent  (sufficiency; later the writer's draft)
        self.reader = HubReader(cfg, n_lang)      # utterance latent -> hub  (minimality; later the read path)
        if cfg.hub_mode == "free":                # the unconstrained anchor exists only where it is used
            self.free_out = nn.Linear(W, d)
            self.free_img = nn.Linear(d, d)
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
    def view(self, z: torch.Tensor, mod: str) -> torch.Tensor:
        """What the hub reads of a modality latent: its content tokens, and the listening tokens when the latent
        carries them (an encoded utterance does; a written one does not)."""
        return hub_view(z, self.visible[mod], self.written[mod])

    def slot_features(self, z: torch.Tensor, mod: str, lang: torch.Tensor) -> torch.Tensor:
        z = self.view(z, mod)
        ctx = self.lat_in(z) + self.lat_pos[: z.shape[1]] + self.mod_emb.weight[self.MODS[mod]] + self.lang_emb(lang).unsqueeze(1)
        h = self.slots.unsqueeze(0).expand(z.shape[0], -1, -1)
        for blk in self.blocks:
            h = blk(h, ctx=ctx)
        return self.norm(h)

    def queries(self, z: torch.Tensor, mod: str, lang: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.q_out(self.slot_features(z, mod, lang)), dim=-1)

    def keys(self, V: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key(V), dim=-1)

    @property
    def tau(self) -> float:
        return self.cfg.hub_temp * self.temp_scale

    def _sequential(self, logits: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
        """Sequential attention with inhibition of return.  logits [..., K, P], null [..., K] ->
        log-probabilities [..., K, P + 1]; the last column is the null option, so every row sums to one."""
        taken = torch.zeros_like(logits[..., 0, :])
        rows = []
        for k in range(logits.shape[-2]):
            la = torch.log_softmax(torch.cat([logits[..., k, :] - self.cfg.hub_inhibit / self.tau * taken,
                                              null[..., k:k + 1]], dim=-1), dim=-1)
            rows.append(la)
            taken = taken + la[..., :-1].exp()
        return torch.stack(rows, -2)

    def _null_logits(self, q: torch.Tensor) -> torch.Tensor:
        if not self.null_enabled:  # first part of phase A: every slot has to look at something
            return torch.full(q.shape[:-1], -1e4, device=q.device, dtype=torch.float32)
        return (q.float() @ F.normalize(self.null_key, dim=-1).float()) / self.tau + self.null_bias

    def ground(self, q: torch.Tensor, keys: torch.Tensor, item: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One pass for everything the grounder needs.

        q [n, K, W] slot queries, keys [B, P, W] patch keys of every image of the batch, item [n] the image of
        each utterance.  Returns
          score [n, B]         how well every image supports the slot sequence (contrastive logits),
          attn  [n, K, P-1]    the attention on the utterance's own image (its hub code is attn @ patches),
          logp  [n, K, P]      its logarithm, with the null option as last column (never clamped, so a loss on
                               it reaches a patch that currently has no attention at all),
        all read off the same sequential, inhibited attention.  Three properties were each needed in practice:
        * patches only - the global token is the image mean; a slot that settles on it makes the hub scene-level;
        * sequential with inhibition of return, and a score computed under that same attention - with the usual
          "best patch per slot, independently" score every slot is fully credited for re-finding one object;
        * a null option (hub token = origin), offered only after the slots have learned their roles, and soft
          attention annealed to sharp.
        """
        cos = torch.einsum("nkw,bpw->nbkp", q, keys[:, :-1]).float()
        null = self._null_logits(q).unsqueeze(1).expand(-1, keys.shape[0], -1)
        logp = self._sequential(cos / self.tau, null)
        attn = logp[..., :-1].exp()
        score = (attn * cos).sum(-1).mean(-1)
        rows = torch.arange(q.shape[0], device=q.device)
        return score, attn[rows, item], logp[rows, item]

    def attend(self, q: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Own-image path only (cache building, figures): q [n, K, W], V [n, P, d] -> hub [n, K, d], attn."""
        logits = torch.einsum("nkw,npw->nkp", q, self.keys(V)[:, :-1]).float() / self.tau
        attn = self._sequential(logits, self._null_logits(q))[..., :-1].exp()
        return torch.einsum("nkp,npd->nkd", attn.to(V.dtype), V[:, :-1]), attn

    def free_code(self, z: torch.Tensor, mod: str, lang: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        """Unconstrained learned anchor (baseline): average of an utterance code and an image code."""
        return 0.5 * (self.free_out(self.slot_features(z, mod, lang)) + self.free_img(scene))

    # ------------------------------------------------------------------ retrieval in hub space
    @staticmethod
    def hub_to_image_scores(hub: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """How well each hub code [n, K, d] is explained by the patches of each gallery image [G, P, d].
        Grounder-free: used to score hub estimates that came out of a backward ODE.  -> [n, G]"""
        h, v = F.normalize(hub.float(), dim=-1), F.normalize(V[:, :-1].float(), dim=-1)
        best = torch.einsum("nkd,gpd->ngkp", h, v).amax(-1)                 # [n, G, K]
        w = hub.float().norm(dim=-1)                                        # empty (null) slots weigh ~0
        return (best * w.unsqueeze(1)).sum(-1) / w.sum(-1, keepdim=True).clamp_min(1e-6)


class _SlotTransformer(nn.Module):
    """n_out learned queries read a set of n_in tokens (shared shape of the Explainer and the Reader)."""

    def __init__(self, cfg, n_lang: int, n_in: int, n_out: int, layers: int):
        super().__init__()
        W, d = cfg.hub_width, cfg.latent_dim
        self.inp = nn.Linear(d, W)
        self.in_pos = nn.Parameter(torch.randn(n_in, W) * 0.02)
        self.mod_emb = nn.Embedding(2, W)
        self.lang_emb = nn.Embedding(n_lang, W)
        self.queries = nn.Parameter(torch.randn(n_out, W) * 0.02)
        self.blocks = nn.ModuleList([Block(W, cfg.hub_heads, ctx_dim=W, dropout=cfg.dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(W)
        self.out = nn.Linear(W, d)

    def forward(self, x: torch.Tensor, mod: int, lang: torch.Tensor, n_out: Optional[int] = None) -> torch.Tensor:
        """``n_out``: how many of the output queries to use (the latents of the modalities can differ in length)."""
        ctx = self.inp(x) + self.in_pos[: x.shape[1]] + self.mod_emb.weight[mod] + self.lang_emb(lang).unsqueeze(1)
        q = self.queries[: n_out].unsqueeze(0).expand(x.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk(q, ctx=ctx)
        return self.out(self.norm(q))


class Explainer(_SlotTransformer):
    """hub code [n, K_h, d] -> expected utterance latent [n, N, d].

    Phase A: sufficiency - the hub code alone must reconstruct the utterance (the t = 0 view of flow matching:
    what the hub does not determine is unexplained variance of the latent).  Afterwards it is the writer's
    *draft*: generation may start from the interpolation between the source and this prediction."""

    def __init__(self, cfg, n_lang: int):
        super().__init__(cfg, n_lang, cfg.hub_slots, cfg.max_tokens, 2)
        # (it explains the tokens the hub reads; what a decoder alone reads is not the hub's to explain)
        self.n_out = {m: cfg.hub_tokens_of(name) for name, m in VisualInterlingua.MODS.items()}

    def forward(self, code: torch.Tensor, mod: int, lang: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(code, mod, lang, self.n_out[mod])


class HubReader(_SlotTransformer):
    """utterance latent [n, N, d] -> hub code [n, K_h, d], with no image.

    Phase A: minimality - the grounded hub code must be predictable from the utterance alone.  Phase B and test
    time: the read path.  Reading is a function (one code per sentence), so it is amortised in one pass;
    writing is a distribution (many realisations per code), so it is a flow.  The reader regresses onto codes
    that do not depend on the speaker, so the hub it returns carries no speaker identity."""

    def __init__(self, cfg, n_lang: int):
        super().__init__(cfg, n_lang, cfg.max_tokens, cfg.hub_slots, cfg.hub_layers)
        self.visible = {m: cfg.hub_tokens_of(m) for m in VisualInterlingua.MODS}
        self.written = {m: cfg.tokens_of(m) for m in VisualInterlingua.MODS}

    def forward(self, z: torch.Tensor, mod: str, lang: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(hub_view(z, self.visible[mod], self.written[mod]), VisualInterlingua.MODS[mod], lang)
