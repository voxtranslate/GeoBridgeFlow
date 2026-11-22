#!/usr/bin/env python
"""Refit ONLY the preview pixel decoder of a finished run, and store it back into the run's checkpoints.

    python scripts/refit_preview.py runs/<exp> --steps 1500

The preview decoder maps the frozen image latent to a thumbnail.  Its input is a frozen projection of frozen
vision features, so it shares no parameter and no gradient with the rest of the model: refitting it changes no
metric, only what `infer` draws in `15_generated_images.png`.  Use it when a run was trained before the
contrast-weighted preview loss existed, or with ``w_pix = 0``-style settings that left the decoder untrained.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geobridge.config import Config  # noqa: E402
from geobridge.data import GeoBridgeData  # noqa: E402
from geobridge.inferencer import load_model  # noqa: E402
from geobridge.losses import preview_loss  # noqa: E402
from geobridge.utils import resolve_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = Config.load(os.path.join(a.run, "config.json"))
    device = resolve_device(cfg.device)
    data = GeoBridgeData(cfg)
    ck_dir = os.path.join(a.run, "checkpoints")
    model, _ = load_model(cfg, data, device, os.path.join(ck_dir, "best.pt"), use_ema=False)
    if not hasattr(model, "pixel") or data.thumbs is None:
        raise SystemExit("this run has no preview decoder (w_pix = 0 or no image writer)")
    model.requires_grad_(False)
    pix = model.pixel.train().requires_grad_(True)
    for m in pix.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    opt = torch.optim.AdamW(pix.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    images = sorted({g["img"] for g in data.groups["train"]})
    rng = np.random.default_rng(cfg.seed)
    for step in range(a.steps):
        idx = rng.choice(images, min(a.batch_size, len(images)), replace=False)
        raw = torch.from_numpy(np.stack([np.asarray(data.vision[i], dtype=np.float32) for i in idx])).to(device)
        target = torch.from_numpy(np.stack([np.asarray(data.thumbs[i]) for i in idx])).to(device).permute(0, 3, 1, 2).float() / 255.0
        loss, l1 = preview_loss(pix(model.image_latent(model.hub.project(raw))).float(), target, cfg.pix_contrast)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"step {step:5d} | weighted {loss.item():.4f} | plain L1 {l1.item():.4f}", flush=True)
    state = {f"pixel.{k}": v.detach().cpu() for k, v in pix.state_dict().items()}
    for name in ("best.pt", "last.pt"):
        path = os.path.join(ck_dir, name)
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            ck["model"].update(state)
            ck["preview_refit"] = {"steps": a.steps, "pix_contrast": cfg.pix_contrast}
            torch.save(ck, path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"preview decoder stored in {path}")


if __name__ == "__main__":
    main()
