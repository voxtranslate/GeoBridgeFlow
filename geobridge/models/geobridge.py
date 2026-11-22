r"""GeoBridge-Flow: the assembled model.

    text  --GBC codec-->  z_txt  --reader-->  \                           /--text writer (flow)-->  z_txt
    speech -tone-gated->  z_aud  --reader-->   }  hub z*  (barycentres   {---speech writer (flow)->  z_aud
    image --frozen ViT->  z_img  --ODE------>  /   of frozen patches)     \--image writer (flow)-->  z_img

Reading is a function: one hub code per sentence.  It is done by the image-free **reader** in a single pass
(or, optionally, by the backward ODE of the modality's flow).  Writing is a distribution: one hub code has
many valid realisations, so every writer is a flow whose source is ``[hub tokens ; private Gaussian tokens]``.
Training-time hub codes come from the grounder and need the paired image; at test time nothing but the
source is needed.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .audio_codec import AudioCodec
from .flow import Spoke
from .hub import VisualInterlingua
from .pixel import PreviewDecoder
from .text_codec import TextCodec


class GeoBridge(nn.Module):
    def __init__(self, cfg, spec: Dict):
        super().__init__()
        self.cfg, self.spec = cfg, spec
        self.langs: List[str] = list(spec["langs"])
        self.lang2id = {l: i for i, l in enumerate(self.langs)}
        n_lang = len(self.langs)
        anchor = spec.get("anchor") or {
            "mean": torch.zeros(spec["vision_dim"]), "basis": torch.zeros(spec["vision_dim"], cfg.latent_dim),
            "scale": torch.ones(())}
        stats = spec.get("audio_stats") or {
            "sem_mean": torch.zeros(spec["sem_dim"]), "sem_std": torch.ones(spec["sem_dim"]),
            "spk_mean": torch.zeros(spec["spk_dim"]), "spk_std": torch.ones(spec["spk_dim"]),
            "f0_mean": torch.zeros(2), "f0_std": torch.ones(2), "mel_mean": 0.0, "mel_std": 1.0}
        self.text = TextCodec(cfg, spec["unit_vocab"], spec["unit_pad"], spec["unit_bytes"], n_lang)
        self.audio = AudioCodec(cfg, spec["sem_dim"], spec["spk_dim"], n_lang, stats)
        self.hub = VisualInterlingua(cfg, spec["vision_dim"], spec["n_vis_tokens"], anchor, n_lang)
        spokes = {"txt": Spoke(cfg, n_lang), "aud": Spoke(cfg, n_lang)}
        if cfg.image_spoke:
            spokes["img"] = Spoke(cfg, 0)
            if cfg.w_pix > 0:
                self.pixel = PreviewDecoder(cfg)
        self.spokes = nn.ModuleDict(spokes)

    # ------------------------------------------------------------------ parameter groups
    def phase_a_modules(self) -> List[nn.Module]:
        mods: List[nn.Module] = [self.text, self.audio, self.hub]
        return mods + ([self.pixel] if hasattr(self, "pixel") else [])

    def phase_b_modules(self) -> List[nn.Module]:
        """Writers and the reader, which keeps training on the cached (now fixed) codes; the explainer too
        when writing starts from its draft.  (The unconstrained 'free' anchor is learned jointly with the flows.)"""
        if self.cfg.hub_mode == "free":
            return [self.spokes, self.hub]
        return [self.spokes, self.hub.reader] + ([self.hub.explainer] if self.cfg.draft_t0 > 0 else [])

    # ------------------------------------------------------------------ modality latents
    def encode(self, mod: str, sub: Dict[str, torch.Tensor], gate_override: Optional[float] = None) -> torch.Tensor:
        if mod == "txt":
            return self.text.encode(sub["units"], sub["unit_mask"])
        return self.audio.encode(sub["sem"], sub["tone"], sub["mask"], sub["lang"], gate_override=gate_override)

    def image_latent(self, V: torch.Tensor) -> torch.Tensor:
        return self.hub.image_latent(V, self.cfg.n_tokens)

    # ------------------------------------------------------------------ training-time hub code of a paired view
    def paired_hub(self, mod: str, z: torch.Tensor, lang: torch.Tensor, V_item: torch.Tensor,
                   anchor_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Hub code of image-paired utterances under the configured anchor (V_item [n, P, d] = their images).
        Used when no contrastive score is needed (latent cache, figures); phase A uses ``hub.ground``."""
        mode = self.cfg.hub_mode
        if mode == "bary":
            return self.hub.attend(self.hub.queries(z, mod, lang), V_item)[0]
        if mode == "global":
            return self.hub.scene_code(V_item)
        if mode == "free":
            return self.hub.free_code(z, mod, lang, self.hub.scene_code(V_item))
        if mode == "text":
            return (anchor_z if anchor_z is not None else z)[:, : self.cfg.hub_slots]
        raise ValueError(mode)

    # ------------------------------------------------------------------ read: modality -> hub
    def read(self, mod: str, z: torch.Tensor, lang: Optional[torch.Tensor] = None, mode: Optional[str] = None,
             steps: Optional[int] = None) -> torch.Tensor:
        """Hub code of a modality latent with nothing but the latent.  ``amortized`` = the reader, one pass;
        ``ode`` = backward flow.  Images have no reader (an image does not say one thing): always the ODE."""
        mode = mode or self.cfg.read_mode
        if mod == "img" or mode == "ode":
            lang = torch.zeros(z.shape[0], dtype=torch.long, device=z.device) if lang is None or mod == "img" else lang
            return self.spokes[mod].read(z, lang, steps=steps)[0]
        return self.hub.reader(z, mod, lang)

    # ------------------------------------------------------------------ write: hub -> modality
    def write(self, mod: str, hub: torch.Tensor, lang: Optional[torch.Tensor] = None, temp: Optional[float] = None,
              steps: Optional[int] = None, private: Optional[torch.Tensor] = None, t0: Optional[float] = None) -> torch.Tensor:
        """Modality latent written from a hub code by the flow.  With ``t0 > 0`` the ODE starts from the
        explainer's one-pass draft (text and speech only)."""
        if lang is None or mod == "img":
            lang = torch.zeros(hub.shape[0], dtype=torch.long, device=hub.device)
        t0 = self.cfg.draft_t0 if t0 is None else t0
        draft = None
        if t0 > 0 and mod != "img":
            draft = self.hub.explainer(hub, self.hub.MODS[mod], lang)
        return self.spokes[mod].write(hub, lang, temp=temp, private=private, steps=steps, draft=draft, t0=t0)

    @torch.no_grad()
    def hub_to_text(self, hub: torch.Tensor, lang_name: str, temp: Optional[float] = None) -> List[str]:
        lang = torch.full((hub.shape[0],), self.lang2id[lang_name], dtype=torch.long, device=hub.device)
        return self.text.generate(self.write("txt", hub, lang, temp=temp), lang)

    @torch.no_grad()
    def hub_to_speech(self, hub: torch.Tensor, lang_name: str, spk: torch.Tensor, f0: torch.Tensor,
                      temp: Optional[float] = None):
        """hub -> (log-mel, frame counts, speech latent, streams).  ``spk, f0`` give the voice (bypass)."""
        lang = torch.full((hub.shape[0],), self.lang2id[lang_name], dtype=torch.long, device=hub.device)
        z = self.write("aud", hub, lang, temp=temp)
        mel, n_frames, streams = self.audio.synthesize_mel(z, lang, spk, f0)
        return mel, n_frames, z, streams

    @torch.no_grad()
    def hub_to_image(self, hub: torch.Tensor, temp: Optional[float] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """hub -> (image latent [B, N, d], preview RGB [B, 3, S, S] or None)."""
        z = self.write("img", hub, temp=temp)
        return z, (self.pixel(z) if hasattr(self, "pixel") else None)


def build_spec(data) -> Dict:
    """Everything the model needs to know about the data; stored in every checkpoint."""
    u = data.unitizer
    return {"langs": list(data.cfg.all_langs), "lang_info": data.lang_info, "vision_dim": data.vision_dim,
            "n_vis_tokens": data.n_vis_tokens, "sem_dim": data.sem_dim, "spk_dim": data.spk_dim,
            "unit_vocab": u.unit_vocab, "unit_pad": u.unit_pad, "unit_bytes": u.m,
            "anchor": {k: v for k, v in data.anchor.items()}, "audio_stats": dict(data.audio.stats)}


def spec_for_checkpoint(spec: Dict) -> Dict:
    """Shapes only (the tensors themselves are model buffers and travel in the state dict)."""
    return {k: v for k, v in spec.items() if k not in ("anchor", "audio_stats")}
