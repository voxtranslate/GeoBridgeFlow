"""Flickr8k-Tri: the real benchmark, assembled from four public Hub repositories (see download.py).

Join keys (verified on the pinned revisions):
    image id            1000268201_693b08cb0e
    English caption k   k-th line of that image in captions.txt            (k = 0..4)
    English speech      audio id  <image id>_<k>                            (Flickr Audio Caption Corpus)
    Yoruba speech/text  filename  S001_<image id>_<k>   (always k = 0)      (YFACC)
    Hindi text          k-th row of that image in flickr8k_hindi.csv

So the Yoruba utterance for an image is a spoken translation of English caption 0 of the same image, which
gives an exact parallel reference for zero-pair Yoruba-speech -> English-text scoring.  The Yoruba transcript
is carried in the manifest but, with the default config, never shown to the model: Yoruba is treated as an
unwritten language and its orthography is used for analysis only.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import io
import os
from typing import Dict, Iterator, List, Tuple

import numpy as np

from .download import download_sources

FLICKR_LANGS = {
    "en": {"tonal": False, "script": "Latin", "written": True, "spoken": True},
    "yo": {"tonal": True, "script": "Latin+tone marks (held out)", "written": False, "spoken": True},
    "hi": {"tonal": False, "script": "Devanagari (machine-translated, silver)", "written": True, "spoken": False},
}
YFACC_SPLIT = {"train": "train", "val": "valid", "test": "test"}


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


class Flickr8kTri:
    def __init__(self, cfg, logger=None) -> None:
        self.cfg, self.logger = cfg, logger
        keys = ["captions", "images_en_speech", "yoruba"] + (["hindi"] if "hi" in cfg.all_langs else [])
        self.dirs = download_sources("flickr8k_yo", cfg.data_root, keys, logger)

    # ------------------------------------------------------------------ manifest
    def manifest(self) -> Dict:
        import pyarrow.parquet as pq

        cfg = self.cfg
        caps: Dict[str, List[str]] = {}
        with open(os.path.join(self.dirs["captions"], "captions.txt"), encoding="utf8") as f:
            next(f)
            for line in f:
                name, cap = line.rstrip("\n").split(",", 1)
                caps.setdefault(name[:-4], []).append(cap.strip().strip('"'))

        # which parquet file holds which image / utterance (ids only; no media is decoded here)
        self.img_file: Dict[str, Tuple[str, int]] = {}
        for fp in sorted(glob.glob(os.path.join(self.dirs["images_en_speech"], "images", "*.parquet"))):
            for r, i in enumerate(pq.read_table(fp, columns=["id"]).column("id").to_pylist()):
                self.img_file[i] = (fp, r)
        self.aud_file: Dict[str, Tuple[str, int]] = {}
        for fp in sorted(glob.glob(os.path.join(self.dirs["images_en_speech"], "audio", "*.parquet"))):
            for r, i in enumerate(pq.read_table(fp, columns=["id"]).column("id").to_pylist()):
                self.aud_file["en/" + i] = (fp, r)
        yo: Dict[str, Dict] = {}
        for fp in sorted(glob.glob(os.path.join(self.dirs["yoruba"], "data", "*.parquet"))):
            split = YFACC_SPLIT[os.path.basename(fp).split("-")[0]]
            t = pq.read_table(fp, columns=["filename", "transcript"]).to_pydict()
            for r, (fn, tr) in enumerate(zip(t["filename"], t["transcript"])):
                img, k = fn.split("_", 1)[1].rsplit("_", 1)
                yo[img] = {"cap": int(k), "text": tr, "split": split}
                self.aud_file[f"yo/{img}_{k}"] = (fp, r)
        hi: Dict[str, List[str]] = {}
        if "hi" in cfg.all_langs:
            with open(os.path.join(self.dirs["hindi"], "flickr8k_hindi.csv"), encoding="utf8") as f:
                rd = csv.reader(f, delimiter=";")
                next(rd)
                for row in rd:
                    if len(row) >= 4:
                        hi.setdefault(row[1][:-4], []).append(row[3].strip())

        ids = sorted(i for i in self.img_file if i in caps)
        if cfg.max_images and cfg.max_images < len(ids):
            ids = sorted(sorted(ids, key=_h)[: cfg.max_images])
        images, utts = [], []
        for n, img in enumerate(ids):
            images.append({"id": img, "official_split": yo[img]["split"] if img in yo else ""})
            for k, cap in enumerate(caps[img][:5]):
                utts.append({"uid": f"en/{img}/{k}", "img": n, "cap": k, "lang": "en", "text": cap,
                             "has_audio": f"en/{img}_{k}" in self.aud_file, "spk": "", "src": f"en/{img}_{k}"})
                if img in hi and k < len(hi[img]) and "hi" in cfg.all_langs:
                    utts.append({"uid": f"hi/{img}/{k}", "img": n, "cap": k, "lang": "hi", "text": hi[img][k],
                                 "has_audio": False, "spk": "", "src": ""})
            if img in yo and "yo" in cfg.all_langs:
                k = yo[img]["cap"]
                utts.append({"uid": f"yo/{img}/{k}", "img": n, "cap": k, "lang": "yo", "text": yo[img]["text"],
                             "has_audio": True, "spk": "yo_S001", "src": f"yo/{img}_{k}"})
        langs = {l: FLICKR_LANGS[l] for l in cfg.all_langs if l in FLICKR_LANGS}
        return {"name": "flickr8k_yo", "langs": langs, "images": images, "utts": utts,
                "meta": {"n_images_total": len(self.img_file)}}

    # ------------------------------------------------------------------ media access
    def image_groups(self, images: List[Dict]) -> List[Dict]:
        by_file: Dict[str, List[Tuple[int, int]]] = {}
        for n, im in enumerate(images):
            fp, r = self.img_file[im["id"]]
            by_file.setdefault(fp, []).append((n, r))
        groups = []
        for fp, rows in sorted(by_file.items()):
            def loader(fp=fp, rows=rows) -> Iterator[Tuple[int, object]]:
                import pyarrow.parquet as pq
                from PIL import Image

                col = pq.read_table(fp, columns=["image"]).column("image")
                for n, r in rows:
                    yield n, Image.open(io.BytesIO(col[r].as_py()["bytes"])).convert("RGB")

            groups.append({"key": os.path.basename(fp), "idx": [n for n, _ in rows], "loader": loader})
        return groups

    def load_image(self, image_id: str):
        import pyarrow.parquet as pq
        from PIL import Image

        fp, r = self.img_file[image_id]
        col = pq.read_table(fp, columns=["image"]).column("image")
        return Image.open(io.BytesIO(col[r].as_py()["bytes"])).convert("RGB")

    def audio_groups(self, utts: List[Dict]) -> List[Dict]:
        by_file: Dict[str, List[Tuple[str, int]]] = {}
        for u in utts:
            if u["has_audio"]:
                fp, r = self.aud_file[u["src"]]
                by_file.setdefault(fp, []).append((u["uid"], r))
        groups = []
        for fp, rows in sorted(by_file.items()):
            def loader(fp=fp, rows=rows) -> Iterator[Tuple[str, np.ndarray, int]]:
                import pyarrow.parquet as pq
                import soundfile as sf

                col = pq.read_table(fp, columns=["audio"]).column("audio")
                for uid, r in rows:
                    wav, sr = sf.read(io.BytesIO(col[r].as_py()["bytes"]), dtype="float32", always_2d=True)
                    yield uid, wav.mean(1), sr

            tag = "yo_" if "yfacc" in fp else "en_"
            groups.append({"key": tag + os.path.basename(fp), "uids": [u for u, _ in rows], "loader": loader})
        return groups
