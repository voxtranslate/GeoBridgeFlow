"""Small shared helpers: seeding, devices, JSON io, logging, parameter accounting."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def get_logger(name: str = "geobridge", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def load_json(path: str) -> Any:
    with open(path, encoding="utf8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def count_named(module: torch.nn.Module, must_contain: Iterable[str]) -> int:
    keys = tuple(must_contain)
    return sum(p.numel() for n, p in module.named_parameters() if any(k in n for k in keys))


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)) and len(batch) and isinstance(batch[0], torch.Tensor):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def lap(self) -> float:
        t = time.time()
        dt, self.t0 = t - self.t0, t
        return dt


class Averager:
    """Running mean of scalar dictionaries (used for tqdm postfixes and epoch summaries)."""

    def __init__(self) -> None:
        self.sum: Dict[str, float] = {}
        self.n: Dict[str, int] = {}

    def add(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            if v is None:
                continue
            v = float(v)
            if v != v:  # NaN
                continue
            self.sum[k] = self.sum.get(k, 0.0) + v
            self.n[k] = self.n.get(k, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {k: self.sum[k] / max(1, self.n[k]) for k in self.sum}

    def reset(self) -> None:
        self.sum.clear()
        self.n.clear()


def fmt_metrics(d: Dict[str, float], keys: Optional[Iterable[str]] = None, nd: int = 3) -> Dict[str, str]:
    keys = list(keys) if keys is not None else list(d)
    return {k: f"{d[k]:.{nd}f}" for k in keys if k in d}


def grid_factor(n: int) -> tuple:
    """Most square (rows, cols) factorisation of n (used to lay out image-latent tokens)."""
    r = int(np.floor(np.sqrt(n)))
    while r > 1 and n % r:
        r -= 1
    return r, n // r


def setup_matplotlib() -> None:
    """Headless backend + a font fallback chain that covers Devanagari and other non-Latin scripts when a
    suitable system font exists (figures label graphemes of every language)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    chain = [f for f in ("DejaVu Sans", "Noto Sans Devanagari", "Lohit Devanagari", "FreeSerif", "FreeSans",
                         "Noto Sans", "Arial Unicode MS") if f in have]
    if chain:
        matplotlib.rcParams["font.family"] = chain
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
