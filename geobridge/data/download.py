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

Licensing note: the two speech mirrors carry no licence metadata on the Hub.  Cite the original corpora and
check their terms before redistributing derived data.
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
