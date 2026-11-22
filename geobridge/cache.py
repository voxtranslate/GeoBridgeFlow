"""Latent cache for the transport phase.

In phase B the codecs and the grounder are frozen, so the latent of every utterance and its grounded hub code
are constants.  They are computed once here and phase B then runs on tensors only: no feature loading, no
codec, no grounder, no image.  Rebuilding after a resume is deterministic and takes a fraction of an epoch.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .losses import GeoBridgeLoss, same_lang_pairs
from .utils import to_device


class LatentCache:
    def __init__(self, model, data, split: str, device: torch.device, loss: GeoBridgeLoss, chunk: int = 96,
                 island: bool = True):
        self.device, cfg = device, model.cfg
        groups = data.groups[split]
        key = "train_views" if split == "train" else "valid_views"
        z = {"txt": [], "aud": []}
        hub = {"txt": [], "aud": []}
        ok = {"txt": [], "aud": []}
        lang = {"txt": [], "aud": []}
        count = {"txt": [0] * len(groups), "aud": [0] * len(groups)}
        img_lat: Dict[int, torch.Tensor] = {}
        scene: Dict[int, torch.Tensor] = {}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for s in range(0, len(groups), chunk):
                part = groups[s: s + chunk]
                batch = to_device(data.collate_groups([{"img": g["img"], "cap": g["cap"], "views": g[key]} for g in part]), device)
                V = model.hub.project(batch["img"])
                lat, sc = model.image_latent(V), model.hub.scene_code(V)
                for j, g in enumerate(part):
                    if g["img"] not in img_lat:
                        img_lat[g["img"]], scene[g["img"]] = lat[j].cpu(), sc[j].cpu()
                zz = {m: model.encode(m, batch[m]) for m in ("txt", "aud") if batch[m] is not None}
                if cfg.hub_mode == "text" and batch.get("pivot") is not None:
                    zz["pivot"] = model.encode("txt", batch["pivot"])
                pairs = same_lang_pairs(batch["txt"], batch["aud"])
                for m in ("txt", "aud"):
                    if m not in zz:
                        continue
                    sub = batch[m]
                    anchor, valid = (None, torch.ones(len(sub["item"]), dtype=torch.bool, device=device))
                    if cfg.hub_mode == "text":
                        anchor, valid = loss.text_anchor(zz, m, sub, pairs, batch.get("pivot"), batch["B"])
                    h = zz[m][:, : cfg.hub_slots] if cfg.hub_mode == "free" else \
                        model.paired_hub(m, zz[m], sub["lang"], V[sub["item"]], anchor)
                    z[m].append(zz[m].float().cpu()); hub[m].append(h.float().cpu())
                    ok[m].append(valid.cpu()); lang[m].append(sub["lang"].cpu())
                    for j in sub["item"].tolist():  # views arrive grouped by caption, in order
                        count[m][s + j] += 1
            self.isl_z: Optional[torch.Tensor] = None
            if island and split == "train" and len(data.island_ids) >= 2:
                zs, ls = [], []
                for s in range(0, len(data.island_ids), chunk):
                    b = to_device(data.collate_island(data.island_ids[s: s + chunk]), device)
                    zs.append(model.encode("aud", b["aud"]).float().cpu()); ls.append(b["aud"]["lang"].cpu())
                self.isl_z, self.isl_lang = torch.cat(zs), torch.cat(ls)
        model.train(was_training)
        self.mods = [m for m in ("txt", "aud") if z[m]]
        self.z = {m: torch.cat(z[m]) for m in self.mods}
        self.hub = {m: torch.cat(hub[m]) for m in self.mods}
        self.ok = {m: torch.cat(ok[m]) for m in self.mods}
        self.lang = {m: torch.cat(lang[m]) for m in self.mods}
        self.start = {m: torch.tensor([0] + count[m]).cumsum(0) for m in self.mods}   # rows of group g: [start[g], start[g+1])
        self.img_of_group = torch.tensor([g["img"] for g in groups])
        imgs = sorted(img_lat)
        self.img_row = {i: n for n, i in enumerate(imgs)}
        self.img_lat = torch.stack([img_lat[i] for i in imgs])
        self.scene = torch.stack([scene[i] for i in imgs])
        self._all_hubs = torch.cat([self.hub[m][self.ok[m]] for m in self.mods]) if self.mods else torch.zeros(0)
        self.n_groups = len(groups)

    def batch(self, group_idx: List[int]) -> Dict:
        """Phase-B batch for a list of caption-group indices (same layout as the collated raw batch)."""
        dev = self.device
        rows_img = torch.tensor([self.img_row[int(self.img_of_group[g])] for g in group_idx])
        out: Dict = {"B": len(group_idx), "img_lat": self.img_lat[rows_img].to(dev), "scene": self.scene[rows_img].to(dev),
                     "txt": None, "aud": None}
        for m in self.mods:
            rows, items = [], []
            for j, g in enumerate(group_idx):
                a, b = int(self.start[m][g]), int(self.start[m][g + 1])
                rows.extend(range(a, b)); items.extend([j] * (b - a))
            if rows:
                r = torch.tensor(rows)
                out[m] = {"z": self.z[m][r].to(dev), "hub": self.hub[m][r].to(dev), "ok": self.ok[m][r].to(dev),
                          "lang": self.lang[m][r].to(dev), "item": torch.tensor(items, device=dev)}
        return out

    def island_batch(self, idx: List[int]) -> Dict:
        r = torch.tensor(idx)
        return {"z": self.isl_z[r].to(self.device), "lang": self.isl_lang[r].to(self.device)}

    def sample_hubs(self, n: int) -> torch.Tensor:
        """Real hub codes of image-paired utterances (any language): the empirical hub distribution."""
        r = torch.randint(len(self._all_hubs), (n,))
        return self._all_hubs[r].to(self.device)
