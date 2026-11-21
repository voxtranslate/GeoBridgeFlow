"""GeoBridge-Flow: the assembled model.

    text  --GBC codec-->  z_txt  <--text spoke-->  \
    speech -tone-gated->  z_aud  <--speech spoke->  }  hub z*  (barycentres of frozen vision tokens)
    image --frozen ViT->  z_img  <--image spoke-->  /

Training-time hub codes come from the grounder (they need the paired image).  Test-time hub codes come from
the *backward ODE* of the source spoke, so no image, no text and no grounder are needed to translate.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .audio_codec import AudioCodec
from .flow import Spoke
from .hub import HubBank, VisualInterlingua
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
        self.spokes = nn.ModuleDict(spokes)
        self.bank = HubBank(cfg.dock_bank_size, cfg.hub_slots, cfg.latent_dim)
        self.free_img = nn.Linear(cfg.latent_dim, cfg.latent_dim)  # only used by the "free" anchor baseline

    # ------------------------------------------------------------------ parameter groups
    def phase_a_modules(self) -> List[nn.Module]:
        return [self.text, self.audio, self.hub, self.free_img]

    def phase_b_modules(self) -> List[nn.Module]:
        mods: List[nn.Module] = [self.spokes]
        if self.cfg.hub_mode == "free":  # the unconstrained anchor is learned jointly with the flows
            mods += [self.hub, self.free_img]
        return mods

    def lora_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if "lora_" in n]

    # ------------------------------------------------------------------ modality latents
    def encode(self, mod: str, sub: Dict[str, torch.Tensor], gate_override: Optional[float] = None) -> torch.Tensor:
        if mod == "txt":
            return self.text.encode(sub["units"], sub["unit_mask"])
        return self.audio.encode(sub["sem"], sub["tone"], sub["mask"], sub["lang"], gate_override=gate_override)

    def image_latent(self, V: torch.Tensor) -> torch.Tensor:
        return self.hub.image_latent(V, self.cfg.n_tokens)

    # ------------------------------------------------------------------ training-time hub codes
    def paired_hub(self, mod: str, z: torch.Tensor, lang: torch.Tensor, V_item: torch.Tensor,
                   pivot_z: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Hub code of each image-paired utterance under the configured anchor.  V_item [n, P, d]."""
        mode, K = self.cfg.hub_mode, self.cfg.hub_slots
        q = self.hub.queries(z, mod, lang)
        hub_b, attn = self.hub.attend(q, V_item)
        out = {"q": q, "attn": attn}
        if mode == "bary":
            out["hub"] = hub_b
        elif mode == "global":
            out["hub"] = self.hub.scene_code(V_item)
        elif mode == "free":
            out["hub"] = 0.5 * (self.hub.free_code(z, mod, lang) + self.free_img(self.hub.scene_code(V_item)))
        elif mode == "text":
            out["hub"] = (pivot_z if pivot_z is not None else z)[:, :K]
        else:
            raise ValueError(mode)
        return out

    # ------------------------------------------------------------------ test-time conversion
    @torch.no_grad()
    def to_hub(self, mod: str, z: torch.Tensor, lang: Optional[torch.Tensor] = None, steps: Optional[int] = None) -> torch.Tensor:
        lang = torch.zeros(z.shape[0], dtype=torch.long, device=z.device) if lang is None or mod == "img" else lang
        return self.spokes[mod].read(z, lang, steps=steps)[0]

    @torch.no_grad()
    def from_hub(self, mod: str, hub: torch.Tensor, lang: Optional[torch.Tensor] = None, temp: Optional[float] = None,
                 steps: Optional[int] = None, private: Optional[torch.Tensor] = None) -> torch.Tensor:
        lang = torch.zeros(hub.shape[0], dtype=torch.long, device=hub.device) if lang is None or mod == "img" else lang
        return self.spokes[mod].write(hub, lang, temp=temp, private=private, steps=steps)

    @torch.no_grad()
    def hub_from_batch(self, mod: str, batch: Dict, steps: Optional[int] = None,
                       gate_override: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(hub estimate, source latent) for the txt / aud sub-batch or the images of a collated batch."""
        if mod == "img":
            z = self.image_latent(self.hub.project(batch["img"]))
            return self.to_hub("img", z, steps=steps), z
        sub = batch[mod]
        z = self.encode(mod, sub, gate_override=gate_override)
        return self.to_hub(mod, z, sub["lang"], steps=steps), z

    @torch.no_grad()
    def hub_to_text(self, hub: torch.Tensor, lang_name: str, temp: Optional[float] = None) -> List[str]:
        lang = torch.full((hub.shape[0],), self.lang2id[lang_name], dtype=torch.long, device=hub.device)
        return self.text.generate(self.from_hub("txt", hub, lang, temp=temp), lang)

    @torch.no_grad()
    def hub_to_speech(self, hub: torch.Tensor, lang_name: str, spk: torch.Tensor, f0: torch.Tensor,
                      temp: Optional[float] = None):
        """hub -> (log-mel, frame counts, speech latent).  ``spk, f0`` give the voice (paralinguistic bypass)."""
        lang = torch.full((hub.shape[0],), self.lang2id[lang_name], dtype=torch.long, device=hub.device)
        z = self.from_hub("aud", hub, lang, temp=temp)
        mel, n_frames, streams = self.audio.synthesize_mel(z, lang, spk, f0)
        return mel, n_frames, z, streams


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
