"""Small shared helpers: seeding, devices, JSON io, logging, parameter accounting."""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
import sys
import time
import warnings
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


def gpu_ids(cfg, device: torch.device) -> list:
    """GPUs a run may use: all visible ones (``cfg.n_gpu = 0``) or the first ``cfg.n_gpu``; [] off CUDA."""
    if device.type != "cuda" or device.index not in (None, 0):
        return [device.index] if device.type == "cuda" else []
    n = torch.cuda.device_count()
    return list(range(min(n, cfg.n_gpu) if cfg.n_gpu > 0 else n))


def worker_kwargs(num_workers: int) -> Dict[str, Any]:
    """DataLoader worker settings.  Workers are forked where the platform allows it: the dataset (memory-mapped
    feature arrays, and classes that may have been defined in a notebook cell) is then shared, not pickled."""
    if num_workers <= 0:
        return {"num_workers": 0}
    fork = "fork" in multiprocessing.get_all_start_methods()
    return {"num_workers": num_workers, **({"multiprocessing_context": "fork"} if fork else {})}


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


def grid_factor(n: int) -> tuple:
    """Most square (rows, cols) factorisation of n (used to lay out image-latent tokens)."""
    r = int(np.floor(np.sqrt(n)))
    while r > 1 and n % r:
        r -= 1
    return r, n // r


class quiet_transformers:
    """Context manager: load a pretrained model without the library's informational notices (a vision tower whose
    config carries text-token ids, a vocoder whose hub config names an older class, ...)."""

    def __enter__(self):
        from transformers.utils import logging as hf_logging

        self._log, self._level = hf_logging, hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        return self

    def __exit__(self, *exc) -> bool:
        self._log.set_verbosity(self._level)
        return False


class OutOfTime(RuntimeError):
    """The session's time budget is used up.  Raised by the trainer right after it has saved the exact step it
    reached, so that running the same command again continues from there."""


_BUDGET: Dict[str, float] = {}


def set_time_budget(hours: float, since: Optional[float] = None) -> None:
    """Training stops cleanly ``hours`` after ``since`` (a ``time.time()`` value; default: now).  0 = no limit.

    Hosted notebooks end a session after a fixed time and may keep nothing of a run that is cut off; a run that
    stops by itself keeps its checkpoints, curves and tables."""
    _BUDGET.clear()
    if hours and hours > 0:
        _BUDGET["deadline"] = (time.time() if since is None else since) + 3600.0 * hours


def time_left() -> float:
    """Seconds until the time budget is used up (infinite when there is none)."""
    return _BUDGET.get("deadline", float("inf")) - time.time()


# A font per script that the default matplotlib font lacks: (a character of the script, font file, source).
_SCRIPT_FONTS = (
    ("\u0915", "NotoSansDevanagari-Regular.ttf",
     "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/fonts/NotoSansDevanagari/hinted/ttf/"
     "NotoSansDevanagari-Regular.ttf"),
)


_FONTS: Dict[str, Any] = {}


def _font_chain(font_dir: str) -> tuple:
    """(font fallback chain, scripts still without a font).  Scans the installed fonts once per process."""
    from matplotlib import font_manager, ft2font

    def covering(char: str) -> list:
        names = []
        for f in font_manager.fontManager.ttflist:
            try:
                if ord(char) in ft2font.FT2Font(f.fname).get_charmap():
                    names.append(f.name)
            except Exception:
                continue
        return names

    chain, missing = ["DejaVu Sans"], []
    for char, fname, url in _SCRIPT_FONTS:
        have = covering(char)
        if not have and font_dir:
            path = os.path.join(font_dir, fname)
            try:
                if not os.path.exists(path):
                    import urllib.request

                    os.makedirs(font_dir, exist_ok=True)
                    urllib.request.urlretrieve(url, path + ".tmp")
                    os.replace(path + ".tmp", path)
                font_manager.fontManager.addfont(path)
                have = [font_manager.FontProperties(fname=path).get_name()]
            except Exception:   # offline, or the mirror moved: boxes instead of glyphs, nothing else changes
                have = []
        if not have:
            missing.append(fname)
        chain += [n for n in have[:1] if n not in chain]
    known = {f.name for f in font_manager.fontManager.ttflist}
    chain += [n for n in ("FreeSerif", "FreeSans", "Noto Sans", "Arial Unicode MS") if n in known and n not in chain]
    return [n for n in chain if n in known], missing


def setup_matplotlib(font_dir: str = "") -> None:
    """Headless backend (outside notebooks) and a font fallback chain that covers every script of the benchmark.

    Figures label graphemes of each language.  Where the system has no font for a script (Kaggle and Colab images
    ship none for Devanagari) and ``font_dir`` is given, one small font file is fetched there, once.  Without a
    font the labels are drawn as boxes; either way the figure is written, and matplotlib's per-glyph warnings are
    silenced."""
    import matplotlib

    if "ipykernel" not in sys.modules:  # in a notebook the inline backend stays; figures are saved and closed
        matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r"Glyph \d+ .*missing from", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"Matplotlib currently does not support .* natively", category=UserWarning)
    if "chain" not in _FONTS or (font_dir and _FONTS["missing"]):
        _FONTS["chain"], _FONTS["missing"] = _font_chain(font_dir)
    if _FONTS["chain"]:
        matplotlib.rcParams["font.family"] = _FONTS["chain"]


def legend_if_any(ax, **kwargs) -> None:
    """``ax.legend`` only when something on the axis has a label (an empty legend raises a warning per figure)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kwargs)
