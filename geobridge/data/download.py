"""Automatic download of the exact public corpora the benchmark is assembled from.

Every source is pinned to the revision that was verified to exist and to join correctly
(image id + caption index) on 2026-11-21.  If a pinned revision disappears the loader falls back to
``main`` and says so.  Nothing here needs an account or a gated-dataset agreement.

Benchmark "flickr8k_yo" = Flickr8k-Tri:
    images ............. 8,091 Flickr8k photos
    en text ............ 40,455 crowd captions (5 / image)                  [Hodosh et al., 2013]
    en speech .......... 40,000 spoken captions, 183 speakers (FACC)        [Harwath & Glass, 2015]
    yo speech (+text) .. 5,981 spoken Yoruba captions, 1 speaker (YFACC)    [Olaleye et al., 2023]
    hi text ............ 40,455 machine-translated Hindi captions (silver)  -> script-robustness branch

Extra hub-language data (``cfg.extra_images``), training only:
    images + en text ... Flickr30k, 31,783 photos with 5 crowd captions each     [Young et al., 2014]
                         8,091 of them ARE the Flickr8k photos, with the same captions.  They are dropped by
                         photo id (flickr8k.py), all of them and whatever their split: otherwise test images and
                         the captions held out for the other languages would come back in through this door.

Extra hub-language SPEECH (``cfg.extra_speech``), training only:
    images + en text + en speech ... SpokenCOCO, the 5,000-image test part: 25,031 captions read aloud by 1,766
                         speakers (Flickr8k's speech has 183)                    [Hsu et al., 2021; CC BY-SA 4.0]
                         The mirror holds audio and images; the caption text and the Flickr address of every
                         image come from the COCO annotations, joined by caption id (in the audio file name)
                         and image id.  A COCO photo that is a Flickr8k photo would be dropped by the same
                         photo-id rule (none of the 5,000 is).

Licensing note: the two Flickr8k speech mirrors and the Flickr30k mirror carry no licence metadata on the Hub; the
photos are under Flickr terms of use.  Cite the original corpora and check their terms before redistributing
derived data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HubSource:
    repo: str
    revision: str
    patterns: Tuple[str, ...]
    note: str


FLICKR8K_YO: Dict[str, HubSource] = {
    "captions": HubSource("shivangibithel/Flickr8k", "89173bc6ee039ac347ced51fb19ea1260d4c4b38",
                          ("captions.txt",), "English captions, 5 per image, in Flickr8k.token order"),
    "images_en_speech": HubSource("deep9539/flickr-audio-image", "eccef427e2398ade609d40c1dcc9b5c30dce8641",
                                  ("images/*.parquet", "audio/*.parquet", "qrels/*.parquet"),
                                  "Flickr8k images + Flickr Audio Caption Corpus (id = <image>_<caption idx>)"),
    "yoruba": HubSource("nolimitsxl/yfacc_yoruba", "bc5607b24083120b1c528f4c8126eca5d89a1485",
                        ("data/*.parquet",), "YFACC: Yoruba spoken captions + tone-marked transcripts"),
    "hindi": HubSource("ArefaMuzaffar/flickr8k_hindi_captions", "1a2a4352bc801562b115c3e5bb970de890f0caf4",
                       ("flickr8k_hindi.csv",), "machine-translated Hindi captions (Devanagari)"),
}

REGISTRY: Dict[str, Dict[str, HubSource]] = {"flickr8k_yo": FLICKR8K_YO}

# Flickr30k as nine parquet shards (image bytes, five captions, file name = Flickr photo id), sorted by photo id.
FLICKR30K = HubSource("lmms-lab/flickr30k", "765d117f3eec816f2bfdc2d73ebb50a6f77b86a4", ("data/test-*.parquet",),
                      "Flickr30k photos + 5 English captions each")
FLICKR30K_SHARDS = 9


def local_dir(data_root: str, repo: str) -> str:
    return os.path.join(data_root, "raw", repo.replace("/", "__"))


def download_sources(dataset: str, data_root: str, keys: List[str] = None, logger=None) -> Dict[str, str]:
    """Download (idempotent) and return {source key: local directory}."""
    from huggingface_hub import snapshot_download

    out: Dict[str, str] = {}
    for key, src in REGISTRY[dataset].items():
        if keys is not None and key not in keys:
            continue
        d = local_dir(data_root, src.repo)
        marker = os.path.join(d, f".complete_{src.revision[:12]}")
        if not os.path.exists(marker):
            if logger:
                logger.info(f"[download] {src.repo}@{src.revision[:8]}  ({src.note})")
            try:
                snapshot_download(src.repo, repo_type="dataset", revision=src.revision,
                                  allow_patterns=list(src.patterns), local_dir=d, max_workers=6)
            except Exception as e:  # pinned commit gone -> latest
                if logger:
                    logger.info(f"[download] pinned revision unavailable ({type(e).__name__}); using 'main'")
                snapshot_download(src.repo, repo_type="dataset", allow_patterns=list(src.patterns),
                                  local_dir=d, max_workers=6)
            with open(marker, "w") as f:
                f.write("ok\n")
        out[key] = d
    return out


# SpokenCOCO, test part (an audio-to-image retrieval set): "queries" = spoken captions, clips of one image together,
# file name <speaker>-<assignment>_<COCO image id>_<COCO caption id>; "corpus" = the images.
SPOKENCOCO = HubSource("whybe-choi/SpokenCOCOA2IRetrieval", "72731c49562b73ea7e1cbd296c40f3924a4e5fad",
                       ("queries/test-*.parquet", "corpus/test-*.parquet"), "SpokenCOCO test part: spoken captions + images")
SPOKENCOCO_AUDIO_SHARDS, SPOKENCOCO_IMAGE_SHARDS = 7, 2
# COCO annotations as two small tables: image id, Flickr address, caption ids, captions.
COCO_CAPTIONS = HubSource("phiyodr/coco2017", "036f3f8291db64d17faad9b09e59dd30bb65c4d7",
                          ("data/train-00000-of-00001-0084e041f1902997.parquet",
                           "data/validation-00000-of-00001-e3c37e369512a3aa.parquet"),
                          "COCO captions by caption id, and the Flickr address of every image")


def download_file(src: HubSource, name: str, data_root: str, logger=None) -> str:
    """Download one file of a pinned source (idempotent) and return its path."""
    from huggingface_hub import hf_hub_download

    d = local_dir(data_root, src.repo)
    path = os.path.join(d, name)
    if not os.path.exists(path):
        if logger:
            logger.info(f"[download] {src.repo}@{src.revision[:8]} {name}  ({src.note})")
        try:
            hf_hub_download(src.repo, name, repo_type="dataset", revision=src.revision, local_dir=d)
        except Exception as e:  # pinned commit gone -> latest
            if logger:
                logger.info(f"[download] pinned revision unavailable ({type(e).__name__}); using 'main'")
            hf_hub_download(src.repo, name, repo_type="dataset", local_dir=d)
    return path


def download_flickr30k(data_root: str, shard: int, logger=None) -> str:
    """Download one shard of the Flickr30k mirror (idempotent) and return its path.  Shards are fetched one by one,
    so a small ``extra_images`` needs one file of 0.5 GB and not all 4.4 GB."""
    from huggingface_hub import hf_hub_download

    src = FLICKR30K
    d = local_dir(data_root, src.repo)
    name = f"data/test-{shard:05d}-of-{FLICKR30K_SHARDS:05d}.parquet"
    path = os.path.join(d, name)
    if not os.path.exists(path):
        if logger:
            logger.info(f"[download] {src.repo}@{src.revision[:8]} {name}  ({src.note})")
        try:
            hf_hub_download(src.repo, name, repo_type="dataset", revision=src.revision, local_dir=d)
        except Exception as e:  # pinned commit gone -> latest
            if logger:
                logger.info(f"[download] pinned revision unavailable ({type(e).__name__}); using 'main'")
            hf_hub_download(src.repo, name, repo_type="dataset", local_dir=d)
    return path
