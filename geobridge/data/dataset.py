"""Dataset assembly, persistent splits, and the three data streams used by the trainer.

Streams
  paired  one item = one caption group (image, caption index) with every *visible* view of it:
          hub-language text + speech, text-only-language text, and island speech only if that utterance is
          one of the bridge pairs.
  island  unpaired speech of the island language(s): no image, no text, no link to anything.
  eval    caption groups of the valid / test split with all views (held-out text is exposed as reference).

Splits are created once and stored in ``splits.json``; later runs load the file, they never re-draw it.
Bridge membership is a stored per-utterance rank in [0, 1): an utterance is a bridge pair iff
rank < bridge_frac, so the 1 % / 5 % / 20 % / 100 % budgets are nested subsets of one another.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..text_units import TextUnitizer
from ..utils import get_logger, load_json, save_json, worker_kwargs
from . import features as feats


class _GroupDataset(Dataset):
    def __init__(self, data: "GeoBridgeData", groups: List[Dict], train: bool) -> None:
        self.data, self.groups, self.train = data, groups, train

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> Dict:
        g = self.groups[i]
        return {"img": g["img"], "cap": g["cap"], "views": g["train_views"] if self.train else g["valid_views"],
                "train": self.train}


class _IslandDataset(Dataset):
    def __init__(self, utt_ids: List[int]) -> None:
        self.ids = utt_ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> int:
        return self.ids[i]


class GeoBridgeData:
    def __init__(self, cfg, logger=None, device: Optional[torch.device] = None, need_mel: bool = True) -> None:
        self.cfg = cfg
        self.log = logger or get_logger()
        device = device or torch.device("cpu")
        os.makedirs(cfg.cache_dir, exist_ok=True)

        if cfg.dataset == "synthworld":
            from .synthworld import SynthWorld

            self.builder = SynthWorld(cfg)
        elif cfg.dataset == "flickr8k_yo":
            from .flickr8k import Flickr8kTri

            self.builder = Flickr8kTri(cfg, self.log)
        else:
            raise ValueError(f"unknown dataset {cfg.dataset}")

        # ---- manifest (built once)
        mp = os.path.join(cfg.cache_dir, "manifest.json")
        if os.path.exists(mp):
            self.manifest = load_json(mp)
            if hasattr(self.builder, "img_file") is False and cfg.dataset == "flickr8k_yo":
                self.builder.manifest()  # rebuild the media locators (cheap: ids only)
        else:
            self.manifest = self.builder.manifest()
            save_json(self.manifest, mp)
        self.images, self.utts = self.manifest["images"], self.manifest["utts"]
        self.lang_info = self.manifest["langs"]
        self.langs = [l for l in cfg.all_langs if l in self.lang_info]
        self.lang2id = {l: i for i, l in enumerate(cfg.all_langs)}

        # ---- splits (created once, then always loaded)
        self.splits = self._load_or_make_splits()
        self.img_split = [self.splits["images"][im["id"]] for im in self.images]

        # ---- frozen-feature caches (computed once)
        vdir = feats.cache_vision(cfg, self.builder, self.manifest, device, self.log)
        self.vision = np.load(os.path.join(vdir, "tokens.npy"), mmap_mode="r")
        self.thumbs = None
        if cfg.w_pix > 0 and cfg.image_spoke:
            self.thumbs = np.load(feats.cache_thumbs(cfg, self.builder, self.manifest, vdir, self.log), mmap_mode="r")
        self.vision_dim, self.n_vis_tokens = int(self.vision.shape[2]), int(self.vision.shape[1])
        adir = feats.cache_audio(cfg, self.builder, self.manifest, self.splits, device, self.log)
        mdir = feats.cache_mel(cfg, self.builder, self.manifest, self.splits, adir, self.log) if need_mel and cfg.w_mel > 0 else None
        self.audio = feats.AudioStore(adir, mdir)
        self.sem_dim, self.spk_dim = int(self.audio.stats["dim"]), int(self.audio.stats["spk_dim"])
        train_imgs = np.array([i for i, s in enumerate(self.img_split) if s == "train"])
        self.anchor = feats.fit_anchor(self.vision, train_imgs, cfg.latent_dim,
                                       os.path.join(vdir, f"anchor_d{cfg.latent_dim}.pt"))
        self.max_frames = int(cfg.max_audio_sec * 25)

        # ---- text units
        bpe_path = None
        if cfg.text_unit == "bpe":
            bpe_path = os.path.join(cfg.cache_dir, f"bpe_{cfg.bpe_vocab}_{'-'.join(cfg.hub_langs)}.json")
            if not os.path.exists(bpe_path):
                texts = [u["text"] for u in self.utts
                         if u["lang"] in cfg.hub_langs and self.img_split[u["img"]] == "train" and u["text"]]
                TextUnitizer.train_bpe(texts, bpe_path, cfg.bpe_vocab)
        self.unitizer = TextUnitizer(cfg.text_unit, cfg.max_graphemes, cfg.max_bytes_per_grapheme,
                                     cfg.max_text_bytes, bpe_path)
        self._build_groups()

    # ------------------------------------------------------------------ splits
    def _load_or_make_splits(self) -> Dict:
        cfg, path = self.cfg, os.path.join(self.cfg.cache_dir, "splits.json")
        if os.path.exists(path):
            sp = load_json(path)
            if all(im["id"] in sp["images"] for im in self.images):
                return sp
            self.log.info("[splits] existing splits.json does not cover the manifest; extending it")
        else:
            sp = {"images": {}, "bridge_rank": {}}
        rng = np.random.default_rng(0)  # fixed: the split is a property of the dataset, not of the run
        free = [im["id"] for im in self.images if im["id"] not in sp["images"] and not im.get("official_split")]
        for im in self.images:
            if im["id"] not in sp["images"] and im.get("official_split"):
                sp["images"][im["id"]] = im["official_split"]
        perm = rng.permutation(len(free))
        a, b, _ = cfg.split_fracs
        n_tr, n_va = int(round(a * len(free))), int(round(b * len(free)))
        for rank, j in enumerate(perm):
            sp["images"][free[j]] = "train" if rank < n_tr else ("valid" if rank < n_tr + n_va else "test")
        island = sorted(u["uid"] for u in self.utts if u["uid"] not in sp["bridge_rank"])
        ranks = rng.permutation(len(island)) / max(1, len(island))
        for uid, r in zip(island, ranks):
            sp["bridge_rank"][uid] = float(r)
        counts: Dict[str, int] = {}
        for s in sp["images"].values():
            counts[s] = counts.get(s, 0) + 1
        sp["counts"], sp["created"] = counts, time.strftime("%Y-%m-%d %H:%M:%S")
        sp["note"] = ("image-level split; official corpus split where one exists, otherwise a fixed random draw. "
                      "bridge_rank < bridge_frac marks the island utterances that keep their image pairing.")
        save_json(sp, path)
        self.log.info(f"[splits] created {path}: {counts}")
        return sp

    # ------------------------------------------------------------------ views
    def _build_groups(self) -> None:
        """Decide, once, which views of every caption the model may see in training.

        strict_nonparallel (default): in the training split a caption group is visible in exactly one
        language - the island language if it owns that caption index, else a text-only language if it owns
        it, else the hub language(s).  Captions are translations of each other across languages, so without
        this rule the corpus would be implicitly parallel through the shared image."""
        cfg = self.cfg
        by_key: Dict[Tuple[int, int], Dict] = {}
        for ui, u in enumerate(self.utts):
            if u["lang"] not in self.lang2id:
                continue
            g = by_key.setdefault((u["img"], u["cap"]), {"img": u["img"], "cap": u["cap"], "utts": {},
                                                         "train_views": [], "all_views": []})
            g["utts"][u["lang"]] = ui
            if u.get("text"):
                g["all_views"].append(("txt", u["lang"], ui))
            if u["has_audio"] and u["uid"] in self.audio.index:
                g["all_views"].append(("aud", u["lang"], ui))

        self.island_ids: List[int] = []
        self.n_bridge = 0
        self.groups: Dict[str, List[Dict]] = {"train": [], "valid": [], "test": []}
        self.pivots: Dict[int, List[Tuple[int, int]]] = {}
        for key in sorted(by_key):
            g = by_key[key]
            split, cap = self.img_split[g["img"]], g["cap"]
            has = lambda m, l: any(v[0] == m and v[1] == l for v in g["all_views"])
            owner = None  # the single language that may see this caption in training (strict mode)
            for l in cfg.island_langs:
                if cap in cfg.island_caps and has("aud", l):
                    owner = l
            if owner is None:
                for l in cfg.text_only_langs:
                    if l not in cfg.hub_langs and cap in cfg.text_only_caps and has("txt", l):
                        owner = l
            for mod, lang, ui in g["all_views"]:
                u = self.utts[ui]
                if cfg.strict_nonparallel and owner is not None and lang != owner:
                    continue
                if lang in cfg.island_langs:
                    if cap not in cfg.island_caps or not has("aud", lang):
                        continue
                    bridge = self.splits["bridge_rank"].get(u["uid"], 1.0) < cfg.bridge_frac
                    if mod == "aud" and bridge:
                        g["train_views"].append((mod, lang, ui))
                        self.n_bridge += int(split == "train")
                    elif mod == "aud" and split == "train" and cfg.island_unpaired:
                        self.island_ids.append(ui)
                    elif mod == "txt" and bridge and cfg.use_island_text:
                        g["train_views"].append((mod, lang, ui))
                elif lang in cfg.hub_langs:
                    g["train_views"].append((mod, lang, ui))
                elif lang in cfg.text_only_langs and mod == "txt" and (cap in cfg.text_only_caps or not cfg.strict_nonparallel):
                    g["train_views"].append((mod, lang, ui))
            # validation-loss views: everything the model could be trained on, never the held-out island text
            g["valid_views"] = [v for v in g["all_views"]
                                if not (v[0] == "txt" and v[1] in cfg.island_langs and not cfg.use_island_text)]
            if split == "train":
                for mod, lang, ui in g["train_views"]:
                    if mod == "txt" and lang in cfg.hub_langs:
                        self.pivots.setdefault(g["img"], []).append((cap, ui))
                if not g["train_views"]:
                    continue
            self.groups[split].append(g)
        self.replay_pool = sorted({ui for g in self.groups["train"] for m, _, ui in g["train_views"] if m == "txt"})
        self.caps_of_img: Dict[int, List[Dict]] = {}
        for g in by_key.values():
            self.caps_of_img.setdefault(g["img"], []).append(g)
        self._rng = np.random.default_rng(0)

    def describe(self) -> Dict:
        out = {"images": {s: self.img_split.count(s) for s in ("train", "valid", "test")},
               "caption_groups": {s: len(g) for s, g in self.groups.items()},
               "bridge_pairs_train": self.n_bridge, "island_unpaired_train": len(self.island_ids),
               "vision_tokens": [self.n_vis_tokens, self.vision_dim], "speech_dim": self.sem_dim,
               "anchor_explained_var": float(self.anchor["explained"])}
        for l in self.langs:
            us = [u for u in self.utts if u["lang"] == l]
            out[f"lang_{l}"] = {"utterances": len(us), "with_audio": sum(u["uid"] in self.audio.index for u in us),
                                **self.lang_info[l]}
        return out

    # ------------------------------------------------------------------ collation
    def _collate_text(self, ids: Sequence[int], items: Sequence[int]) -> Optional[Dict]:
        if not ids:
            return None
        out = self.unitizer.collate([self.utts[i]["text"] for i in ids])
        out["lang"] = torch.tensor([self.lang2id[self.utts[i]["lang"]] for i in ids])
        out["item"] = torch.tensor(list(items))
        out["utt"] = torch.tensor(list(ids))
        return out

    def _collate_audio(self, ids: Sequence[int], items: Sequence[int]) -> Optional[Dict]:
        if not ids:
            return None
        recs = [self.audio.get(self.utts[i]["uid"], self.max_frames, self.cfg.mel_fps) for i in ids]
        T, n = max(r["sem"].shape[0] for r in recs), len(recs)
        sem = torch.zeros(n, T, self.sem_dim)
        tone = torch.zeros(n, T, 4)
        mask = torch.zeros(n, T, dtype=torch.bool)
        for j, r in enumerate(recs):
            t = r["sem"].shape[0]
            sem[j, :t], tone[j, :t], mask[j, :t] = torch.from_numpy(r["sem"]), torch.from_numpy(r["tone"]), True
        out = {"sem": sem, "tone": tone, "mask": mask,
               "spk": torch.from_numpy(np.stack([r["spk"] for r in recs])),
               "f0": torch.from_numpy(np.stack([r["f0"] for r in recs])),
               "lang": torch.tensor([self.lang2id[self.utts[i]["lang"]] for i in ids]),
               "item": torch.tensor(list(items)), "utt": torch.tensor(list(ids))}
        if "mel" in recs[0]:
            lens = [r["mel"].shape[0] for r in recs]
            mel = torch.zeros(n, max(lens), self.cfg.n_mels)
            for j, r in enumerate(recs):
                mel[j, : lens[j]] = torch.from_numpy(r["mel"])
            out["mel"], out["mel_len"] = mel, torch.tensor(lens)
        return out

    def collate_groups(self, batch: Sequence[Dict]) -> Dict:
        img = torch.from_numpy(np.stack([np.asarray(self.vision[b["img"]], np.float32) for b in batch]))
        t_ids, t_items, a_ids, a_items = [], [], [], []
        for j, b in enumerate(batch):
            for mod, _, ui in b["views"]:
                (t_ids if mod == "txt" else a_ids).append(ui)
                (t_items if mod == "txt" else a_items).append(j)
        out = {"B": len(batch), "img": img, "img_idx": torch.tensor([b["img"] for b in batch]),
               "cap": torch.tensor([b["cap"] for b in batch]),
               "txt": self._collate_text(t_ids, t_items), "aud": self._collate_audio(a_ids, a_items)}
        train = bool(batch[0].get("train"))
        if self.thumbs is not None and train:
            out["thumb"] = torch.from_numpy(np.stack([np.asarray(self.thumbs[b["img"]]) for b in batch]))
        if self.cfg.text_replay > 0 and train and self.replay_pool:
            # text is cheap: the text codec also sees extra captions that take no part in grounding
            pick = self._rng.integers(len(self.replay_pool), size=self.cfg.text_replay)
            out["replay"] = self._collate_text([self.replay_pool[i] for i in pick], [-1] * len(pick))
        if self.cfg.hub_mode == "text":  # text-anchor baseline: one hub-language caption of the SAME IMAGE,
            piv = []                      # never the same caption index (that would be a parallel sentence)
            for b in batch:
                cands = [ui for cap, ui in self.pivots.get(b["img"], []) if cap != b["cap"]]
                piv.append(cands[int(self._rng.integers(len(cands)))] if cands else -1)
            keep = [j for j, ui in enumerate(piv) if ui >= 0]
            out["pivot"] = self._collate_text([piv[j] for j in keep], keep)
        return out

    def collate_island(self, ids: Sequence[int]) -> Dict:
        return {"B": len(ids), "img": None, "txt": None, "aud": self._collate_audio(list(ids), list(range(len(ids))))}

    # ------------------------------------------------------------------ loaders
    def paired_loader(self, split: str = "train", shuffle: bool = True, batch_size: Optional[int] = None) -> DataLoader:
        ds = _GroupDataset(self, self.groups[split], train=(split == "train"))
        return DataLoader(ds, batch_size=batch_size or self.cfg.batch_size, shuffle=shuffle, drop_last=shuffle,
                          collate_fn=self.collate_groups, **worker_kwargs(self.cfg.num_workers if shuffle else 0))

    def eval_groups(self, split: str, max_items: int = 0, require: Sequence[Tuple[str, str]] = ()) -> List[Dict]:
        """Deterministic subset of caption groups that contain all required (modality, language) views."""
        gs = [g for g in self.groups[split]
              if all(any(v[0] == m and v[1] == l for v in g["all_views"]) for m, l in require)]
        if max_items and len(gs) > max_items:
            idx = np.random.default_rng(0).permutation(len(gs))[:max_items]
            gs = [gs[i] for i in sorted(idx)]
        return gs

    def eval_batches(self, groups: List[Dict], batch_size: int = 64):
        for i in range(0, len(groups), batch_size):
            part = groups[i:i + batch_size]
            yield part, self.collate_groups([{"img": g["img"], "cap": g["cap"], "views": g["all_views"]} for g in part])

    def references(self, img: int, lang: str) -> List[str]:
        """All captions of an image in a language (multi-reference scoring)."""
        return [self.utts[g["utts"][lang]]["text"] for g in self.caps_of_img.get(img, [])
                if lang in g["utts"] and self.utts[g["utts"][lang]].get("text")]
