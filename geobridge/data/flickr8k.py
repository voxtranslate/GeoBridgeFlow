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

Extra hub-language speech (``cfg.extra_speech``): the test part of SpokenCOCO, 5,000 COCO photos whose captions
were read aloud (25,031 clips, 1,766 speakers), for training only.  The clip name carries the speaker, the COCO
image id and the COCO caption id; the caption text and the Flickr address of the photo come from the COCO
annotations.  The Flickr8k photo-id rule below applies to these photos as well.

Extra hub-language data (``cfg.extra_images``): Flickr30k photos with their English captions, as text, for
training only.  Flickr30k CONTAINS the Flickr8k photos (file name = the Flickr photo id that Flickr8k ids start
with) together with the very same captions, so every Flickr30k photo whose id is a Flickr8k photo id is dropped:
whatever its Flickr8k split (test images must not come back as training data) and also when the Flickr8k image
is not part of the current subset (its captions 0 and 4 belong to Yoruba and Hindi in the strict protocol).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import io
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .download import (COCO_CAPTIONS, FLICKR30K, FLICKR30K_SHARDS, SPOKENCOCO, SPOKENCOCO_AUDIO_SHARDS,
                       SPOKENCOCO_IMAGE_SHARDS, download_file, download_flickr30k, download_sources, local_dir)

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

    # ------------------------------------------------------------------ extra hub-language image-text pairs
    def flickr8k_photo_ids(self) -> set:
        """Flickr photo id of EVERY Flickr8k image (the part of the image id before the underscore), read from the
        caption file itself so that it does not depend on the subset in use."""
        ids = set()
        with open(os.path.join(self.dirs["captions"], "captions.txt"), encoding="utf8") as f:
            next(f)
            for line in f:
                ids.add(line.split(",", 1)[0].rsplit(".", 1)[0].split("_")[0])
        return ids

    def extra_manifest(self, n: int, speech: bool = False, first: Optional[int] = None) -> Dict:
        """The first ``n`` (all if n < 0) Flickr30k photos that are not Flickr8k photos, with their captions;
        with ``speech``, SpokenCOCO photos with their captions and the recordings of them."""
        import pyarrow.parquet as pq

        cfg, H = self.cfg, self.cfg.hub_langs[0]
        if H != "en":
            raise ValueError("the extra data is English: it needs 'en' as the hub language")
        if speech:
            return self._spokencoco_manifest(n)
        own = self.flickr8k_photo_ids()
        images, utts, dropped, too_long = [], [], 0, 0
        for shard in range(FLICKR30K_SHARDS):
            fp = download_flickr30k(cfg.data_root, shard, self.logger)
            t = pq.read_table(fp, columns=["filename", "caption"]).to_pydict()
            for row, (fn, caps) in enumerate(zip(t["filename"], t["caption"])):
                pid = fn.rsplit(".", 1)[0]
                if pid in own:
                    dropped += 1
                    continue
                caps = [(k, " ".join(c.split())) for k, c in enumerate(caps[:5])]
                keep = [(k, c) for k, c in caps if c and len(c.encode("utf8")) <= cfg.max_text_bytes]
                too_long += len(caps) - len(keep)
                if not keep:
                    continue
                images.append({"id": f"f30k_{pid}", "official_split": "train", "extra": True,
                               "src": [os.path.basename(fp), row]})
                utts += [{"uid": f"{H}/f30k_{pid}/{k}", "img": len(images) - 1, "cap": k, "lang": H, "text": c,
                          "has_audio": False, "spk": "", "src": ""} for k, c in keep]
                if 0 < n <= len(images):
                    break
            if 0 < n <= len(images):
                break
        assert not {im["id"][5:] for im in images} & own
        return {"name": "flickr30k_extra", "images": images, "utts": utts,
                "meta": {"source": f"{FLICKR30K.repo}@{FLICKR30K.revision[:12]}", "shards_read": shard + 1,
                         "dropped_flickr8k_photos": dropped, "dropped_captions_over_max_text_bytes": too_long}}

    def _spokencoco_manifest(self, n: int) -> Dict:
        """SpokenCOCO test part: images, their captions as text, and one recording of every caption.

        Clips of one image lie together in the audio files, so a subset of ``n`` images needs only the first
        audio file(s).  Captions longer than ``max_text_bytes`` are left out with their recordings."""
        import pyarrow.parquet as pq

        cfg, H = self.cfg, self.cfg.hub_langs[0]
        text, flickr = {}, {}
        for name in COCO_CAPTIONS.patterns:   # caption id -> text, image id -> Flickr address
            t = pq.read_table(download_file(COCO_CAPTIONS, name, cfg.data_root, self.logger),
                              columns=["image_id", "flickr_url", "ids", "captions"]).to_pydict()
            for iid, url, ids, caps in zip(t["image_id"], t["flickr_url"], t["ids"], t["captions"]):
                flickr[int(iid)] = url or ""
                text.update({int(c): s for c, s in zip(ids, caps)})
        where: Dict[int, List] = {}           # image id -> [image file, row]
        for k in range(SPOKENCOCO_IMAGE_SHARDS):
            name = f"corpus/test-{k:05d}-of-{SPOKENCOCO_IMAGE_SHARDS:05d}.parquet"
            ids = pq.read_table(download_file(SPOKENCOCO, name, cfg.data_root, self.logger), columns=["id"]).column("id").to_pylist()
            where.update({int(i): [name, r] for r, i in enumerate(ids)})
        own = self.flickr8k_photo_ids()
        photo = lambda iid: flickr.get(iid, "").rsplit("/", 1)[-1].split("_")[0]
        clips: Dict[int, List] = {}           # image id -> [(caption id, speaker, audio file, row)], in file order
        dropped, no_image, too_long, no_text = set(), set(), 0, 0
        for k in range(SPOKENCOCO_AUDIO_SHARDS):
            name = f"queries/test-{k:05d}-of-{SPOKENCOCO_AUDIO_SHARDS:05d}.parquet"
            ids = pq.read_table(download_file(SPOKENCOCO, name, cfg.data_root, self.logger), columns=["id"]).column("id").to_pylist()
            for r, cid in enumerate(ids):
                head, iid, cap = cid.rsplit("_", 2)
                iid, cap = int(iid), int(cap)
                if photo(iid) in own or iid not in where:
                    (dropped if photo(iid) in own else no_image).add(iid)
                    continue
                s = " ".join(text.get(cap, "").split())
                if not s or len(s.encode("utf8")) > cfg.max_text_bytes:
                    too_long, no_text = too_long + bool(s), no_text + (not s)
                    continue
                clips.setdefault(iid, []).append((cap, head.split("-")[0], s, name, r))
            if 0 < n <= len(clips) - 1:       # the last image of a file may continue in the next one
                break
        images, utts = [], []
        for iid in list(clips)[: n if n > 0 else None]:
            images.append({"id": f"coco_{iid}", "official_split": "train", "extra": True, "src": where[iid]})
            for j, (cap, spk, s, name, r) in enumerate(sorted(clips[iid])):
                utts.append({"uid": f"{H}/coco_{iid}/{j}", "img": len(images) - 1, "cap": j, "lang": H, "text": s,
                             "has_audio": True, "spk": f"coco_{spk}", "src": [name, r]})
        assert not {photo(int(im["id"][5:])) for im in images} & own
        return {"name": "spokencoco_extra", "images": images, "utts": utts,
                "meta": {"source": f"{SPOKENCOCO.repo}@{SPOKENCOCO.revision[:12]} + {COCO_CAPTIONS.repo}@{COCO_CAPTIONS.revision[:12]}",
                         "speakers": len({u["spk"] for u in utts}), "dropped_flickr8k_photos": len(dropped),
                         "clips_without_image": len(no_image),
                         "dropped_captions_over_max_text_bytes": too_long, "captions_without_text": no_text}}

    def _row_group_parts(self, fp: str, rows: List[Tuple[object, int]], per_part: int) -> List[Tuple[List[int], Dict]]:
        """Split the wanted (key, row) pairs of a parquet file by row group: [(row groups of a part, {group: [(key, row in group)]})]."""
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(fp).metadata
        starts = np.cumsum([0] + [meta.row_group(i).num_rows for i in range(meta.num_row_groups)])
        want: Dict[int, List[Tuple[object, int]]] = {}
        for key, r in rows:
            g = int(np.searchsorted(starts, r, side="right") - 1)
            want.setdefault(g, []).append((key, r - int(starts[g])))
        keys = sorted(want)
        return [(keys[a: a + per_part], want) for a in range(0, len(keys), per_part)]

    def _extra_file(self, name: str) -> str:
        """Local path of a file of the extra sources (downloaded if it is not there)."""
        if name.startswith(("queries/", "corpus/")):
            return download_file(SPOKENCOCO, name, self.cfg.data_root, self.logger)
        fp = os.path.join(local_dir(self.cfg.data_root, FLICKR30K.repo), "data", name)
        return fp if os.path.exists(fp) else download_flickr30k(self.cfg.data_root, int(name.split("-")[1]), self.logger)

    def extra_image_groups(self, images: List[Dict]) -> List[Dict]:
        """Loader groups of the extra photos: a few parquet row groups each, so that memory stays bounded and an
        interrupted feature extraction resumes from the last finished group."""
        by_file: Dict[str, List[Tuple[int, int]]] = {}
        for n, im in enumerate(images):
            by_file.setdefault(im["src"][0], []).append((n, int(im["src"][1])))
        groups = []
        for name, rows in sorted(by_file.items()):
            fp = self._extra_file(name)
            for part, want in self._row_group_parts(fp, rows, 8 if name.startswith("test-") else 2):

                def loader(fp=fp, part=part, want=want) -> Iterator[Tuple[int, object]]:
                    import pyarrow.parquet as pq
                    from PIL import Image

                    pf = pq.ParquetFile(fp)
                    for g in part:
                        col = pf.read_row_group(g, columns=["image"]).column("image")
                        for n, r in want[g]:
                            yield n, Image.open(io.BytesIO(col[r].as_py()["bytes"])).convert("RGB")

                groups.append({"key": f"{name}_rg{part[0]:04d}", "idx": [n for g in part for n, _ in want[g]],
                               "loader": loader})
        return groups

    def extra_audio_groups(self, utts: List[Dict]) -> List[Dict]:
        """Loader groups of the extra recordings: one parquet row group each (about 700 clips)."""
        by_file: Dict[str, List[Tuple[str, int]]] = {}
        for u in utts:
            if u["has_audio"]:
                by_file.setdefault(u["src"][0], []).append((u["uid"], int(u["src"][1])))
        groups = []
        for name, rows in sorted(by_file.items()):
            fp = self._extra_file(name)
            for part, want in self._row_group_parts(fp, rows, 1):

                def loader(fp=fp, part=part, want=want) -> Iterator[Tuple[str, np.ndarray, int]]:
                    import pyarrow.parquet as pq
                    import soundfile as sf

                    pf = pq.ParquetFile(fp)
                    for g in part:
                        col = pf.read_row_group(g, columns=["audio"]).column("audio")
                        for uid, r in want[g]:
                            wav, sr = sf.read(io.BytesIO(col[r].as_py()["bytes"]), dtype="float32", always_2d=True)
                            yield uid, wav.mean(1), sr

                groups.append({"key": f"coco_{os.path.basename(name)}_rg{part[0]:04d}",
                               "uids": [uid for g in part for uid, _ in want[g]], "loader": loader})
        return groups

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
