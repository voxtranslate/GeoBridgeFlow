"""Captions of the benchmark's training images written by a pretrained image captioner.

The premise of image-pivoted translation is that a picture can be described in the hub language by anyone, also by
a model that has seen a great many pictures.  Oneata & Kamper (2024) train Yoruba speech -> English text on nothing
but such captions, and find that several *sampled* captions per image are what works (fewer than three hurt; beam
search gives one generic sentence).  Here they are extra hub-language text views of images the benchmark already
has: they are grounded like any caption, the text codec reads them, and image-sibling transport writes them from
the code read out of the island's speech.  Only training images are captioned; nothing is added to validation or
test, and no sentence of any other language is involved.

Generated once and cached as JSON; the cache is never regenerated.
"""
from __future__ import annotations

import os
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from ..utils import load_json, quiet_transformers, save_json


def _tag(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name)


def tidy(caption: str) -> str:
    """One caption in the benchmark's style: single spaces, capital first letter, a final ' .'."""
    s = " ".join(caption.replace("\n", " ").split()).strip(" .")
    return (s[0].upper() + s[1:] + " .") if s else ""


class Captioner:
    """A captioning model of the transformers library, called through its own class (the ``image-to-text`` pipeline
    is gone from transformers 5).  GIT and BLIP checkpoints are supported; sampling is multinomial."""

    def __init__(self, name: str, device: torch.device, max_new_tokens: int = 32) -> None:
        from transformers import AutoProcessor

        self.device, self.max_new_tokens = device, max_new_tokens
        with quiet_transformers():
            self.proc = AutoProcessor.from_pretrained(name)
            if "blip" in name.lower():
                from transformers import BlipForConditionalGeneration as Model
            else:
                from transformers import AutoModelForCausalLM as Model
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            self.model = Model.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, images: List, k: int, seed: int) -> List[List[str]]:
        """k sampled captions for every image (the random stream is this call's own: training is not disturbed)."""
        state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        try:
            px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            out = self.model.generate(pixel_values=px, do_sample=True, num_beams=1, num_return_sequences=k,
                                      max_new_tokens=self.max_new_tokens)
            text = self.proc.batch_decode(out, skip_special_tokens=True)
        finally:
            torch.set_rng_state(state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return [text[i * k: (i + 1) * k] for i in range(len(images))]


def generated_captions(cfg, builder, images: List[Dict], img_split: List[str], device: torch.device, logger) -> Dict[str, List[str]]:
    """{image id: its generated captions} for the training images of the benchmark (cached)."""
    if not cfg.captioner or cfg.gen_captions <= 0:
        return {}
    if cfg.hub_langs[0] != "en":
        raise ValueError("the captioners write English: the hub language must be 'en'")
    path = os.path.join(cfg.cache_dir, f"captions_{_tag(cfg.captioner)}_k{cfg.gen_captions}_b{cfg.max_text_bytes}.json")
    if os.path.exists(path):
        return load_json(path)["captions"]
    if not hasattr(builder, "image_groups"):
        raise ValueError(f"dataset '{cfg.dataset}' has no images a captioner could describe")
    todo = [im for im, s in zip(images, img_split) if s == "train"]
    parts = path[:-5] + "_parts"
    os.makedirs(parts, exist_ok=True)
    model, out, dropped = None, {}, 0
    groups = builder.image_groups(todo)
    for gi, grp in enumerate(tqdm(groups, desc=f"captions ({cfg.captioner})", dynamic_ncols=True)):
        part = os.path.join(parts, _tag(grp["key"]) + ".json")
        if os.path.exists(part):
            out.update(load_json(part))
            continue
        if model is None:
            model = Captioner(cfg.captioner, device)
        res: Dict[str, List[str]] = {}
        buf, ids = [], []

        def flush() -> None:
            nonlocal dropped
            if not buf:
                return
            for iid, caps in zip(ids, model(buf, cfg.gen_captions, seed=1000003 * cfg.seed + 7919 * gi + len(res))):
                keep, seen = [], set()
                for c in map(tidy, caps):
                    ok = c and len(c.split()) >= 4 and len(c.encode("utf-8")) <= cfg.max_text_bytes and c.lower() not in seen
                    dropped += int(not ok)
                    if ok:
                        keep.append(c)
                        seen.add(c.lower())
                res[iid] = keep
            buf.clear()
            ids.clear()

        for n, img in grp["loader"]():
            buf.append(img)
            ids.append(todo[n]["id"])
            if len(buf) >= max(1, cfg.feat_batch_size // 2):
                flush()
        flush()
        save_json(res, part)
        out.update(res)
    n_caps = sum(len(v) for v in out.values())
    save_json({"captioner": cfg.captioner, "per_image": cfg.gen_captions, "decoding": "multinomial sampling",
               "images": len(out), "captions_kept": n_caps, "dropped": dropped, "captions": out}, path)
    for fp in os.listdir(parts):
        os.remove(os.path.join(parts, fp))
    os.rmdir(parts)
    logger.info(f"[cache] {n_caps} generated captions for {len(out)} training images ({cfg.captioner}, sampled; "
                f"{dropped} empty, too short, too long or repeated ones dropped) -> {path}")
    return out
