"""Latent cache for the transport phase.

In phase B the codecs and the grounder are frozen, so the latent of every utterance and its grounded hub code
are constants.  They are computed once here and phase B then runs on tensors only: no feature loading, no
codec, no grounder, no image.  Rebuilding after a resume is deterministic and takes a fraction of an epoch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .losses import GeoBridgeLoss, same_lang_pairs
from .text_units import PAD
from .utils import to_device


def time_keep(mask: torch.Tensor, share: float, gen: torch.Generator, spans: int = 2) -> torch.Tensor:
    """SpecAugment-style time masking: mask [B, T] of valid frames -> keep [B, T] in {0, 1}, with ``spans`` random
    spans of up to ``share`` of each utterance hidden.  Drawn from ``gen`` so that a rebuilt cache is the same."""
    B, T = mask.shape
    n = mask.sum(1, keepdim=True).float().cpu()
    pos = torch.arange(T).unsqueeze(0)
    keep = torch.ones(B, T)
    for _ in range(spans):
        width = torch.rand(B, 1, generator=gen) * share * n
        start = torch.rand(B, 1, generator=gen) * (n - width).clamp_min(0)
        keep = keep * ((pos < start) | (pos >= start + width)).float()
    return keep.to(mask.device)


class LatentCache:
    def __init__(self, model, data, split: str, device: torch.device, loss: GeoBridgeLoss, chunk: int = 96,
                 island: bool = True):
        cfg = model.cfg
        n_aug = cfg.read_aug if split == "train" else 0   # augmented encodings of speech, for the reader only
        z_aug: List[torch.Tensor] = []
        groups = data.groups[split]
        key = "train_views" if split == "train" else "valid_views"
        z = {"txt": [], "aud": []}
        utt_aud: List[torch.Tensor] = []     # utterance of every speech row (its frames are fetched for the speaking decoder)
        hub = {"txt": [], "aud": []}
        ok = {"txt": [], "aud": []}
        lang = {"txt": [], "aud": []}
        count = {"txt": [0] * len(groups), "aud": [0] * len(groups)}
        byte_rows: List[torch.Tensor] = []   # byte targets of the text views (for the decoder's write-latent exposure)
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
                if n_aug and batch["aud"] is not None:
                    a, gen = batch["aud"], torch.Generator().manual_seed(1_000_003 * cfg.seed + s)
                    z_aug.append(torch.stack([model.audio.encode(a["sem"], a["tone"], a["mask"], a["lang"],
                                                                 keep=time_keep(a["mask"], cfg.read_aug_span, gen))
                                              for _ in range(n_aug)], 1).to(torch.float16).cpu())
                if cfg.hub_mode == "text" and batch.get("pivot") is not None:
                    zz["pivot"] = model.encode("txt", batch["pivot"])
                pairs = same_lang_pairs(batch["txt"], batch["aud"])
                hh, vv = {}, {}
                for m in ("txt", "aud"):
                    if m not in zz:
                        continue
                    sub = batch[m]
                    anchor, vv[m] = (None, torch.ones(len(sub["item"]), dtype=torch.bool, device=device))
                    if cfg.hub_mode == "text":
                        anchor, vv[m] = loss.text_anchor(zz, m, sub, pairs, batch.get("pivot"), batch["B"])
                    hh[m] = zz[m][:, : cfg.hub_slots] if cfg.hub_mode == "free" else \
                        model.paired_hub(m, zz[m], sub["lang"], V[sub["item"]], anchor)
                if cfg.hub_mode == "bary" and cfg.canonical_text and len(pairs[0]) and len(hh) == 2:
                    # canonical code of a caption: where a transcript exists (same language), speech takes the
                    # text-grounded code - the cleaner of the two views - as reader target and writer source
                    hh["aud"][pairs[1]] = hh["txt"][pairs[0]]
                for m in hh:
                    sub = batch[m]
                    z[m].append(zz[m].float().cpu()); hub[m].append(hh[m].float().cpu())
                    ok[m].append(vv[m].cpu()); lang[m].append(sub["lang"].cpu())
                    if m == "txt":
                        byte_rows.extend(sub["bytes"].cpu().to(torch.int16))
                    else:
                        utt_aud.append(sub["utt"].cpu())
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
        self.z_aug = torch.cat(z_aug) if z_aug else None   # [speech views, n_aug, N, d]
        self.utt_aud = torch.cat(utt_aud) if utt_aud else None
        self.data = data
        self.n_frames_batch = cfg.dec_aud_batch if (split == "train" and cfg.w_dec_aud > 0 and model.audio.mode == "extra") else 0
        self.hub = {m: torch.cat(hub[m]) for m in self.mods}
        self.ok = {m: torch.cat(ok[m]) for m in self.mods}
        self.lang = {m: torch.cat(lang[m]) for m in self.mods}
        self.start = {m: torch.tensor([0] + count[m]).cumsum(0) for m in self.mods}   # rows of group g: [start[g], start[g+1])
        self.bytes = torch.nn.utils.rnn.pad_sequence(byte_rows, batch_first=True, padding_value=PAD) if byte_rows else None
        self.img_of_group = torch.tensor([g["img"] for g in groups])
        imgs = sorted(img_lat)
        self.img_row = {i: n for n, i in enumerate(imgs)}
        self.img_lat = torch.stack([img_lat[i] for i in imgs])
        self.scene = torch.stack([scene[i] for i in imgs])
        self._all_hubs = torch.cat([self.hub[m][self.ok[m]] for m in self.mods]) if self.mods else torch.zeros(0)
        self._rows = {m: self.ok[m].nonzero().squeeze(1) for m in self.mods if self.ok[m].any()}
        n_own = data.n_own_groups if split == "train" else len(groups)   # extra groups come after the own ones
        self.own_rows = {m: int(self.start[m][n_own]) for m in self.mods}
        self.n_groups = len(groups)
        # image siblings: for every caption group, the text rows of the OTHER captions of its image
        self.sib_of_group: List[torch.Tensor] = []
        weak = [l for l in data.langs if l not in cfg.hub_langs]
        self.sib_langs = {"all": None, "none": set(), "weak": {data.lang2id[l] for l in weak}}[cfg.sib_views]
        if split == "train" and cfg.w_sib > 0 and cfg.sib_views != "none" and "txt" in self.mods:
            by_img: Dict[int, List[Tuple[int, int]]] = {}
            st = self.start["txt"]
            for g, grp in enumerate(groups):
                for r in range(int(st[g]), int(st[g + 1])):
                    if bool(self.ok["txt"][r]):
                        by_img.setdefault(grp["img"], []).append((g, r))
            self.sib_of_group = [torch.tensor([r for g2, r in by_img.get(grp["img"], []) if g2 != g], dtype=torch.long)
                                 for g, grp in enumerate(groups)]

    def batch(self, group_idx: List[int]) -> Dict:
        """Phase-B batch for a list of caption-group indices (same layout as the collated raw batch).
        Tensors are returned on the CPU; the training replica that consumes the batch moves it to its GPU."""
        rows_img = torch.tensor([self.img_row[int(self.img_of_group[g])] for g in group_idx])
        out: Dict = {"B": len(group_idx), "img_lat": self.img_lat[rows_img], "scene": self.scene[rows_img],
                     "txt": None, "aud": None}
        for m in self.mods:
            rows, items = [], []
            for j, g in enumerate(group_idx):
                a, b = int(self.start[m][g]), int(self.start[m][g + 1])
                rows.extend(range(a, b)); items.extend([j] * (b - a))
            if rows:
                r = torch.tensor(rows)
                out[m] = {"z": self.z[m][r], "hub": self.hub[m][r], "ok": self.ok[m][r], "lang": self.lang[m][r],
                          "item": torch.tensor(items)}
                if m == "aud" and self.z_aug is not None:   # what the reader reads: the clean latent or a masked one
                    k = torch.randint(self.z_aug.shape[1] + 1, (len(r),))
                    aug = self.z_aug[r, (k - 1).clamp_min(0)].to(out[m]["z"].dtype)
                    out[m]["z_read"] = torch.where((k > 0).view(-1, 1, 1), aug, out[m]["z"])
                if m == "txt" and self.bytes is not None:
                    b = self.bytes[r].long()
                    out[m]["bytes"] = b[:, : int((b != PAD).sum(1).max())]   # trim the padding to this batch
                if m == "aud" and self.n_frames_batch and self.utt_aud is not None:
                    # the real frames of a few of these utterances: targets of the speaking decoder's exposure
                    good = out[m]["ok"].nonzero().squeeze(1)
                    pick_f = good[torch.randperm(len(good))[: self.n_frames_batch]]
                    if len(pick_f):
                        raw = self.data._collate_audio([int(self.utt_aud[r[int(p)]]) for p in pick_f], list(range(len(pick_f))))
                        out[m]["frames"] = {"idx": pick_f, **{k: raw[k] for k in ("sem", "tone", "mask", "spk", "f0")}}
                if self.sib_of_group:   # one random sibling caption per view of a language that gets them
                    lang_ok = torch.ones(len(r), dtype=torch.bool) if self.sib_langs is None else \
                        torch.tensor([int(l) in self.sib_langs for l in out[m]["lang"].tolist()])
                    pick = torch.full((len(r),), -1, dtype=torch.long)
                    for n, j in enumerate(items):
                        cand = self.sib_of_group[group_idx[j]]
                        if bool(lang_ok[n]) and len(cand):
                            pick[n] = cand[int(torch.randint(len(cand), (1,)))]
                    if bool((pick >= 0).any()):
                        s = pick.clamp_min(0)
                        out[m]["sib_ok"] = pick >= 0
                        out[m]["sib_z"], out[m]["sib_lang"] = self.z["txt"][s], self.lang["txt"][s]
                        if self.bytes is not None:
                            sb = self.bytes[s].long()
                            out[m]["sib_bytes"] = sb[:, : max(1, int((sb[pick >= 0] != PAD).sum(1).max()))]
        return out

    def island_batch(self, idx: List[int]) -> Dict:
        r = torch.tensor(idx)
        return {"z": self.isl_z[r], "lang": self.isl_lang[r]}

    def sample_hubs(self, n: int) -> torch.Tensor:
        """Real hub codes of image-paired utterances (any language): the empirical hub distribution."""
        return self._all_hubs[torch.randint(len(self._all_hubs), (n,))]

    def sample_views(self, mod: str, n: int, own_only: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        """n random image-paired views of one modality: latents, their hub codes and languages.

        ``own_only``: draw from the benchmark's own captions, not from the extra hub-language ones.  A uniform
        draw over everything would be almost all hub language, and the text writer of another written language
        would hardly ever be relayed to."""
        if mod not in self._rows:
            return None
        rows = self._rows[mod]
        if own_only:
            rows = rows[rows < self.own_rows[mod]]
        r = rows[torch.randint(len(rows), (n,))]
        return {"z": self.z[mod][r], "hub": self.hub[mod][r], "lang": self.lang[mod][r]}
