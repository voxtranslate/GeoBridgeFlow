"""Trainer: two-phase schedule in one resumable run.

    phase A  (steps [0, phase_a_steps))      codecs, grounder, explainer, reader   -> representation
    phase B  (steps [phase_a_steps, total))  writers (flows), reader, back-translation, on cached latents

Checkpointing
    checkpoints/step_XXXXXXXX.pt   every ``save_every`` steps (last ``keep_last`` kept, milestones kept forever)
    checkpoints/last.pt            most recent state; training resumes from it automatically
    checkpoints/phase_a.pt         end of phase A (shared by phase-B ablations)
    checkpoints/best.pt            best validation score in phase B
A checkpoint stores model, EMA, both optimisers, AMP scaler, RNG states, the full history and the exact
position inside the epoch (epoch index + batch offset of both data streams).  The epoch order is a pure
function of (seed, epoch), so resuming from any step of any epoch continues the same data stream.

Several GPUs
    One optimisation step consumes one batch per GPU (``cfg.step_batches``): every GPU runs the complete
    forward pass of its own batch (codecs, grounder, losses) and the gradients are averaged, i.e. the step sees
    ``n_gpu x batch_size`` caption groups.  See :class:`StepReplica`.
"""
from __future__ import annotations

import glob
import math
import os
import random
import shutil
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .cache import LatentCache
from .config import Config
from .data import GeoBridgeData
from .data.dataset import _GroupDataset, _IslandDataset
from .losses import GeoBridgeLoss
from .metrics import Evaluator
from .models import GeoBridge, build_spec, spec_for_checkpoint
from .utils import (Averager, OutOfTime, count_named, count_params, get_logger, gpu_ids, resolve_device, save_json,
                    seed_everything, time_left, to_device, worker_kwargs)

TQDM_KEYS_A = ("loss", "txt_ce", "txt_acc", "sem_mse", "mel_l1", "ground", "ground_acc", "unexplained", "read", "seek_txt", "pix_l1", "agree")
TQDM_KEYS_B = ("loss", "cfm_txt", "cfm_aud", "cfm_img", "read_txt", "read_aud", "dec_ce", "dec_acc", "bt_write", "bt_read", "bt_relay", "bt_shift", "dock")
# Settings that may differ between a checkpoint and the run that resumes from it (everything else must match).
RESUME_FREE = {"exp_name", "out_root", "data_root", "device", "amp", "num_workers", "torch_threads", "n_gpu", "step_batches",
               "feat_batch_size", "log_every", "eval_every", "save_every", "keep_last", "milestone_every", "eval_max_items",
               "phase_b_steps", "init_from", "train_only", "ode_steps", "ode_solver", "sample_temp", "eval_temp", "read_mode",
               "draft_t0"}


class ResumableBatchSampler(Sampler):
    """Batches of a per-epoch permutation that depends only on (seed, epoch); can start mid-epoch.

    ``extra_pool`` (indices of the bridge pairs) contributes ``n_extra`` items to every batch, drawn with a
    generator seeded by (seed, epoch, batch), so the stream is still a pure function of its position."""

    def __init__(self, n: int, batch_size: int, seed: int, extra_pool: Optional[List[int]] = None, n_extra: int = 0):
        self.n, self.seed = n, seed
        self.pool = list(extra_pool or [])
        self.k = min(n_extra, batch_size // 2) if self.pool else 0
        self.bs = batch_size - self.k
        self.epoch, self.offset = 0, 0

    def set_state(self, epoch: int, offset: int) -> None:
        self.epoch, self.offset = epoch, offset

    def __len__(self) -> int:
        return self.n // self.bs

    def __iter__(self) -> Iterator[List[int]]:
        perm = np.random.default_rng(self.seed * 100003 + self.epoch).permutation(self.n)
        for b in range(self.offset, len(self)):
            batch = perm[b * self.bs: (b + 1) * self.bs].tolist()
            if self.k:
                rng = np.random.default_rng([self.seed, self.epoch, b])
                batch += [self.pool[i] for i in rng.integers(len(self.pool), size=self.k)]
            yield batch


class EMA:
    def __init__(self, params: Dict[str, torch.nn.Parameter], decay: float):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in params.items()}
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, params: Dict[str, torch.nn.Parameter], step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for k, p in params.items():
            self.shadow[k].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def swap_in(self, params: Dict[str, torch.nn.Parameter]) -> None:
        self.backup = {k: p.detach().clone() for k, p in params.items()}
        for k, p in params.items():
            p.copy_(self.shadow[k])

    @torch.no_grad()
    def swap_out(self, params: Dict[str, torch.nn.Parameter]) -> None:
        for k, p in params.items():
            p.copy_(self.backup[k])
        self.backup = {}


def _plain(v):
    """Tuples and lists compare equal (a config read back from JSON holds lists)."""
    return list(v) if isinstance(v, (tuple, list)) else v


def settings_diff(old: Dict, new: Dict, free: Iterable[str] = RESUME_FREE) -> Dict[str, Tuple]:
    """Fields in which two configurations (as dicts) describe different runs: {field: (old, new)}.
    Fields absent from ``old`` (written by an earlier version of the code) count with their default."""
    old = {**Config().to_dict(), **old}
    return {k: (old.get(k), new[k]) for k in new if k not in free and _plain(old.get(k)) != _plain(new[k])}


class StepReplica(nn.Module):
    """The forward half of one optimisation step, in a form ``nn.DataParallel`` can run on every GPU.

    DataParallel scatters tensors, but a batch here is a nested dict of ragged sub-batches (and, in phase B,
    slices of the latent cache).  So one batch is prepared per GPU and the only thing that is scattered is an
    index: the replica that receives ``[i]`` takes the i-th prepared batch, moves it to its own device and
    computes the whole loss there, under its own autocast context.  Losses are gathered into a vector; logs are
    left in a slot of ``job``, which all replicas share by reference."""

    def __init__(self, model: GeoBridge, loss: GeoBridgeLoss, amp: bool):
        super().__init__()
        self.model = model
        self.loss, self.amp = loss, amp
        self.job: Dict = {}

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        i, dev, job = int(idx[0]), idx.device, self.job
        batch, island = to_device(job["batches"][i], dev), to_device(job["islands"][i], dev)
        with torch.autocast(device_type=dev.type, enabled=self.amp):
            if job["phase"] == "A":
                loss, logs = self.loss.phase_a(self.model, batch, island)
            else:
                loss, logs = self.loss.phase_b(self.model, batch, island, job["progress"], job["bank"])
        job["logs"][i] = logs
        return loss.reshape(1)


def chunked(it: Iterable, n: int) -> Iterator[List]:
    """Consecutive lists of up to n items."""
    buf: List = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


class Trainer:
    def __init__(self, cfg: Config, data: Optional[GeoBridgeData] = None, resume: bool = True):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "curves"), exist_ok=True)
        self.log = get_logger(f"gb.{cfg.exp_name}", os.path.join(self.run_dir, "train.log"))
        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        seed_everything(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.data = data or GeoBridgeData(cfg, self.log, self.device)
        self.spec = build_spec(self.data)
        self.model = GeoBridge(cfg, self.spec).to(self.device)
        self.loss = GeoBridgeLoss(cfg, self.model.lang2id)
        self.evaluator = Evaluator(cfg, self.data, self.device)
        self.amp = bool(cfg.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.gpus = gpu_ids(cfg, self.device)
        self.replica = StepReplica(self.model, self.loss, self.amp)
        self.parallel = nn.DataParallel(self.replica, device_ids=self.gpus) if len(self.gpus) > 1 else None
        self.step_batches = cfg.step_batches or max(1, len(self.gpus))

        a_params = [p for m in self.model.phase_a_modules() for p in m.parameters()]
        b_named = {f"{i}.{n}": p for i, m in enumerate(self.model.phase_b_modules()) for n, p in m.named_parameters()}
        self.b_named = b_named
        no_decay = lambda n, p: p.dim() < 2 or "lora_" in n or "emb" in n
        gate = [self.model.audio.gate_logit]
        self.opt_a = torch.optim.AdamW(
            [{"params": [p for p in a_params if p is not gate[0]], "weight_decay": cfg.weight_decay},
             {"params": gate, "weight_decay": 0.0, "fixed_lr": cfg.lr_gate}], lr=cfg.lr, betas=(0.9, 0.98))
        self.opt_b = torch.optim.AdamW(
            [{"params": [p for n, p in b_named.items() if not no_decay(n, p)], "weight_decay": cfg.weight_decay},
             {"params": [p for n, p in b_named.items() if no_decay(n, p)], "weight_decay": 0.0}],
            lr=cfg.lr_flow, betas=(0.9, 0.98))
        self.ema = EMA(b_named, cfg.ema_decay)

        self.train_ds = _GroupDataset(self.data, self.data.groups["train"], train=True)
        bridge_idx = [i for i, g in enumerate(self.data.groups["train"])
                      if any(l in cfg.island_langs for _, l, _ in g["train_views"])]
        self.sampler = ResumableBatchSampler(len(self.train_ds), cfg.batch_size, cfg.seed, bridge_idx, cfg.bridge_per_batch)
        self.isl_sampler = None
        if len(self.data.island_ids) >= 2:
            bs = min(cfg.island_batch_size, len(self.data.island_ids))
            self.isl_sampler = ResumableBatchSampler(len(self.data.island_ids), bs, cfg.seed + 7)
        self.step, self.epoch, self.offset = 0, 0, 0
        self.isl_epoch, self.isl_offset = 0, 0
        self.cache: Dict[str, LatentCache] = {}
        self.history: Dict[str, List[Dict]] = {"train": [], "valid": []}
        self.best = {"score": -1e9, "step": -1}
        self.best_a = {"value": float("inf"), "step": -1}   # phase-A state with the best validation hub losses
        self._describe()
        if resume and not self._resume() and cfg.init_from:   # a resume under other settings raises here
            self._init_from(cfg.init_from)
        cfg.save(os.path.join(self.run_dir, "config.json"))
        save_json(self.data.describe(), os.path.join(self.run_dir, "data.json"))
        self._apply_train_only()

    # ------------------------------------------------------------------ bookkeeping
    def _describe(self) -> None:
        m, cfg = self.model, self.cfg
        total = count_params(m)
        lora = count_named(m.spokes, ["lora_"])
        n_lang = max(1, len(m.langs))
        per_lang = lora / n_lang + sum(count_params(s.vel.lang_emb) for s in m.spokes.values()) / n_lang
        spokes = count_params(m.spokes)
        self.param_report = {"total": total, "text_codec": count_params(m.text), "audio_codec": count_params(m.audio),
                             "hub": count_params(m.hub), "reader": count_params(m.hub.reader),
                             "explainer": count_params(m.hub.explainer), "spokes": spokes,
                             "per_language_flow_params": per_lang,
                             "per_language_pct_of_spokes": 100.0 * per_lang / max(1, spokes - lora)}
        self.log.info(f"[model] {total / 1e6:.2f}M params | text {self.param_report['text_codec'] / 1e6:.2f}M | "
                      f"audio {self.param_report['audio_codec'] / 1e6:.2f}M | hub {self.param_report['hub'] / 1e6:.2f}M | "
                      f"spokes {spokes / 1e6:.2f}M | per-language adapter {per_lang / 1e3:.1f}k "
                      f"({self.param_report['per_language_pct_of_spokes']:.2f}% of the shared flow backbone)")
        d = self.data.describe()
        self.log.info(f"[data] groups {d['caption_groups']} | bridge pairs {d['bridge_pairs_train']} | "
                      f"unpaired island {d['island_unpaired_train']} | device {self.device}"
                      + (f" x{len(self.gpus)} (DataParallel)" if len(self.gpus) > 1 else "")
                      + f" | {self.step_batches} batch(es) of {cfg.batch_size} per step | amp {self.amp}")
        save_json(self.param_report, os.path.join(self.run_dir, "params.json"))

    @property
    def phase(self) -> str:
        return "A" if self.step < self.cfg.phase_a_steps else "B"

    def _lr(self, step_in_phase: int, phase_len: int, base: float) -> float:
        cfg = self.cfg
        warm = min(1.0, (step_in_phase + 1) / max(1, cfg.warmup_steps))
        prog = min(1.0, step_in_phase / max(1, phase_len))
        return base * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    def _set_phase_mode(self) -> None:
        m = self.model
        if self.phase == "A":
            m.train()
            for mod in m.phase_a_modules():
                mod.requires_grad_(True)
        else:
            for mod in m.phase_a_modules():
                mod.eval()
                mod.requires_grad_(False)
            self._ensure_cache()
            for mod in m.phase_b_modules():
                mod.train()
                mod.requires_grad_(True)
        self._apply_train_only()

    def _ensure_cache(self) -> None:
        """Phase B runs on cached latents and hub codes: the frozen codecs and the grounder are run exactly
        once more, here.  Deterministic, so it is simply rebuilt after a resume."""
        if self.cache:
            return
        t0 = time.time()
        self._set_schedules()
        self.cache = {"train": LatentCache(self.model, self.data, "train", self.device, self.loss),
                      "valid": LatentCache(self.model, self.data, "valid", self.device, self.loss, island=False)}
        c = self.cache["train"]
        self.log.info(f"[cache] latents and hub codes of {sum(len(v) for v in c.z.values())} paired views"
                      f"{'' if c.isl_z is None else f' and {len(c.isl_z)} unpaired island utterances'} "
                      f"cached in {time.time() - t0:.1f}s; phase B no longer runs the codecs or the grounder")

    def _apply_train_only(self) -> None:
        """``train_only="lora:<lang>"``: adding a language later - only that language's LoRA rows and
        language embeddings receive updates (the shared field is frozen, so nothing can be forgotten)."""
        spec = self.cfg.train_only
        if not spec.startswith("lora:"):
            return
        lang = self.model.lang2id[spec.split(":")[1]]
        for n, p in self.model.named_parameters():
            keep = "spokes" in n and ("lora_" in n or "lang_emb" in n)
            p.requires_grad_(keep and self.phase == "B")
            if keep and p.requires_grad and not hasattr(p, "_gb_hook"):
                mask = torch.zeros(p.shape[0], *([1] * (p.dim() - 1)), device=p.device)
                mask[lang] = 1.0
                p.register_hook(lambda g, mask=mask: g * mask)
                p._gb_hook = True

    # ------------------------------------------------------------------ checkpoints
    def _state(self) -> Dict:
        return {"model": self.model.state_dict(), "ema": self.ema.shadow, "opt_a": self.opt_a.state_dict(),
                "opt_b": self.opt_b.state_dict(), "scaler": self.scaler.state_dict(), "step": self.step,
                "epoch": self.epoch, "offset": self.offset, "isl_epoch": self.isl_epoch,
                "isl_offset": self.isl_offset, "history": self.history, "best": self.best, "best_a": self.best_a,
                "cfg": self.cfg.to_dict(), "spec": spec_for_checkpoint(self.spec),
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}}

    def save_checkpoint(self, name: Optional[str] = None) -> str:
        path = os.path.join(self.ckpt_dir, name or f"step_{self.step:08d}.pt")
        tmp = path + ".tmp"
        torch.save(self._state(), tmp)
        os.replace(tmp, path)
        if name is None:
            shutil.copyfile(path, os.path.join(self.ckpt_dir, "last.pt"))
            steps = sorted(glob.glob(os.path.join(self.ckpt_dir, "step_*.pt")))
            removable = [p for p in steps[:-self.cfg.keep_last]
                         if int(os.path.basename(p)[5:13]) % max(1, self.cfg.milestone_every)]
            for p in removable:
                os.remove(p)
        return path

    def _check_same_run(self, ck: Dict, path: str) -> None:
        """Resuming continues a run; it must not silently continue it under different settings."""
        diff = settings_diff(ck.get("cfg", {}), self.cfg.to_dict())
        if diff:
            shown = ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in list(diff.items())[:8])
            raise RuntimeError(
                f"{path} was written by a run with different settings ({shown}{', ...' if len(diff) > 8 else ''}). "
                f"Resuming would mix the two.  Use another exp_name for the new settings, or delete {self.run_dir}.")

    def load_checkpoint(self, path: str, weights_only: bool = False) -> Dict:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if not weights_only:
            self._check_same_run(ck, path)
        own = self.model.state_dict()
        state = {k: v for k, v in ck["model"].items() if k not in own or own[k].shape == v.shape}
        skipped = [k for k in ck["model"] if k not in state]  # e.g. a different LoRA rank in an ablation
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected or skipped:
            self.log.info(f"[ckpt] non-strict load: {len(missing)} missing, {len(unexpected)} unexpected, "
                          f"{len(skipped)} shape-mismatched keys")
        if weights_only:
            return ck
        for k, v in ck["ema"].items():
            if k in self.ema.shadow and self.ema.shadow[k].shape == v.shape:
                self.ema.shadow[k] = v.to(self.device)
        self.opt_a.load_state_dict(ck["opt_a"])
        self.opt_b.load_state_dict(ck["opt_b"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step, self.epoch, self.offset = ck["step"], ck["epoch"], ck["offset"]
        self.isl_epoch, self.isl_offset = ck.get("isl_epoch", 0), ck.get("isl_offset", 0)
        self.history, self.best = ck["history"], ck["best"]
        self.best_a = ck.get("best_a", self.best_a)
        try:
            torch.set_rng_state(ck["rng"]["torch"])
            np.random.set_state(ck["rng"]["numpy"])
            random.setstate(ck["rng"]["python"])
        except Exception:
            pass
        return ck

    def _resume(self) -> bool:
        last = os.path.join(self.ckpt_dir, "last.pt")
        cands = [last] if os.path.exists(last) else sorted(glob.glob(os.path.join(self.ckpt_dir, "step_*.pt")))[-1:]
        if not cands:
            return False
        self.load_checkpoint(cands[0])
        self.log.info(f"[resume] {cands[0]} -> step {self.step} (epoch {self.epoch}, batch {self.offset}, phase {self.phase})")
        return True

    def resume_from(self, path: str) -> None:
        """Resume from an explicit checkpoint (any step of any epoch)."""
        self.load_checkpoint(path)
        self.log.info(f"[resume] {path} -> step {self.step} (epoch {self.epoch}, batch {self.offset})")

    def _init_from(self, path: str) -> None:
        ck = self.load_checkpoint(path, weights_only=True)
        if ck["step"] >= ck["cfg"].get("phase_a_steps", 0) and not self.cfg.train_only:
            self.step = self.cfg.phase_a_steps  # representation phase already done: go straight to phase B
            self.history["train"] = [h for h in ck["history"]["train"] if h["step"] <= self.step]
            self.history["valid"] = [h for h in ck["history"]["valid"] if h["step"] <= self.step]
        elif self.cfg.train_only:
            self.step = self.cfg.phase_a_steps
            for k, v in ck["ema"].items():
                if k in self.ema.shadow and self.ema.shadow[k].shape == v.shape:
                    self.ema.shadow[k] = v.to(self.device)
        self.log.info(f"[init] weights from {path}; starting at step {self.step} (phase {self.phase})")

    # ------------------------------------------------------------------ data streams
    def _paired_loader(self):
        self.sampler.set_state(self.epoch, self.offset)
        if self.phase == "B":  # cached latents: a batch is assembled by indexing tensors, no loader needed
            return (self.cache["train"].batch(idx) for idx in self.sampler)
        return DataLoader(self.train_ds, batch_sampler=self.sampler, collate_fn=self.data.collate_groups,
                          pin_memory=self.device.type == "cuda", **worker_kwargs(self.cfg.num_workers))

    def _island_stream(self) -> Optional[Iterator[Dict]]:
        if self.isl_sampler is None:
            return None

        def gen():
            ds = _IslandDataset(self.data.island_ids)
            while True:
                phase = self.phase
                self.isl_sampler.set_state(self.isl_epoch, self.isl_offset)
                if phase == "B":
                    it = (self.cache["train"].island_batch(idx) for idx in self.isl_sampler)
                else:
                    it = iter(DataLoader(ds, batch_sampler=self.isl_sampler, num_workers=0,
                                         collate_fn=self.data.collate_island))
                finished = True
                for b in it:
                    self.isl_offset += 1
                    yield b
                    if self.phase != phase:  # the representation phase ended: switch to cached batches
                        finished = False
                        break
                if finished:
                    self.isl_epoch, self.isl_offset = self.isl_epoch + 1, 0

        return gen()

    def _set_schedules(self) -> None:
        """Step-dependent switches of phase A ("structure first, sparsity later").  Pure functions of the step,
        so they need no state in the checkpoint: dense stage = gates held open, no null slot, soft attention
        annealed to sharp; afterwards gates and slots are released and priced."""
        cfg, m = self.cfg, self.model
        release = cfg.gate_release_frac * cfg.phase_a_steps
        dense = self.step < release
        m.audio.gate_forced_open, m.hub.null_enabled = dense, not dense
        frac = min(1.0, self.step / max(1.0, release))
        m.hub.temp_scale = cfg.hub_temp_start ** (1.0 - frac)  # geometric anneal: start -> 1

    # ------------------------------------------------------------------ one optimisation step
    def train_step(self, batches: List[Dict], islands: List[Optional[Dict]]) -> Dict[str, float]:
        """One optimiser update from ``len(batches)`` batches: one per GPU, run in parallel when there are
        several GPUs (otherwise one after the other, which is plain gradient accumulation)."""
        cfg = self.cfg
        if self.phase == "A":
            opt, lr = self.opt_a, self._lr(self.step, cfg.phase_a_steps, cfg.lr)
        else:
            opt, lr = self.opt_b, self._lr(self.step - cfg.phase_a_steps, cfg.phase_b_steps, cfg.lr_flow)
        for g in opt.param_groups:
            g["lr"] = g.get("fixed_lr", lr)   # the scalar tonality gates keep a constant, much larger step
        self._set_schedules()
        opt.zero_grad(set_to_none=True)
        n = len(batches)
        self.replica.job = {"batches": batches, "islands": islands, "logs": [None] * n, "phase": self.phase,
                            "progress": (self.step - cfg.phase_a_steps) / max(1, cfg.phase_b_steps),
                            "bank": self.cache.get("train")}
        if self.parallel is not None and n > 1:
            losses = self.parallel(torch.arange(n, device=self.device))
        else:
            losses = torch.cat([self.replica(torch.tensor([i], device=self.device)) for i in range(n)])
        loss = losses.mean()
        logs = Averager()
        for l in self.replica.job["logs"]:
            logs.add(l)
        logs = logs.mean()
        self.replica.job = {}
        if not torch.isfinite(loss):
            self.log.info(f"[warn] non-finite loss at step {self.step}; step skipped")
            return logs
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
        logs["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        self.scaler.step(opt)
        self.scaler.update()
        if self.phase == "B":
            self.ema.update(self.b_named, self.step - cfg.phase_a_steps)
        logs["lr"] = lr
        return logs

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, split: str = "valid", max_batches: int = 20, metrics: bool = True) -> Dict[str, float]:
        cfg, m = self.cfg, self.model
        was_training = m.training
        m.eval()
        self._set_schedules()
        use_ema = self.phase == "B" and self.step > cfg.phase_a_steps
        if use_ema:
            self.ema.swap_in(self.b_named)
        avg = Averager()
        if self.phase == "A":
            loader = self.data.paired_loader(split, shuffle=False, batch_size=cfg.batch_size)
            n_batches = len(loader)
        else:
            c = self.cache[split]
            loader = (c.batch(list(range(i, min(i + cfg.batch_size, c.n_groups)))) for i in range(0, c.n_groups, cfg.batch_size))
            n_batches = math.ceil(c.n_groups / cfg.batch_size)
        gen_state = torch.get_rng_state()
        torch.manual_seed(1234)  # same noise / time draws at every validation -> comparable curves
        n_val = min(n_batches, max_batches)
        # a nested bar that refreshes prints one line per refresh in notebook logs: draw it at the start and at the end
        pbar = tqdm(loader, total=n_val, desc=f"valid [{self.phase}]", leave=False, dynamic_ncols=True, mininterval=3600)
        for i, batch in enumerate(pbar):
            if i >= max_batches:
                break
            batch = to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                if self.phase == "A":
                    _, logs = self.loss.phase_a(m, batch, None)
                else:
                    _, logs = self.loss.phase_b(m, batch, None, 1.0)
            avg.add(logs)
            if i == n_val - 1:
                pbar.set_postfix({k: f"{v:.3f}" for k, v in avg.mean().items() if k in TQDM_KEYS_A + TQDM_KEYS_B}, refresh=False)
        out = avg.mean()
        if metrics and self.phase == "B":
            ms = self.evaluator.evaluate(m, split, cfg.eval_max_items, quick=True)
            out.update(ms)
        torch.set_rng_state(gen_state)
        if use_ema:
            self.ema.swap_out(self.b_named)
        m.train(was_training)
        self._set_phase_mode()
        return out

    # ------------------------------------------------------------------ main loop
    def fit(self) -> Dict[str, float]:
        cfg = self.cfg
        total = cfg.total_steps
        if self.step >= total:
            self.log.info(f"[fit] run already complete ({self.step}/{total} steps)")
            return self.history["valid"][-1] if self.history["valid"] else {}
        island = self._island_stream()
        avg, t0 = Averager(), time.time()
        last_val: Dict[str, float] = {}
        while self.step < total:
            self._set_phase_mode()
            loader = self._paired_loader()
            n_ep, k = len(self.sampler), self.step_batches
            pbar = tqdm(chunked(loader, k), initial=self.offset // k, total=math.ceil(n_ep / k), dynamic_ncols=True,
                        mininterval=0.5, desc=f"epoch {self.epoch} | phase {self.phase} | step {self.step}/{total}")
            waited = time.time()
            for batches in pbar:
                phase_before = self.phase
                got = time.time()
                logs = self.train_step(batches, [next(island) if island is not None else None for _ in batches])
                logs.update(t_data=got - waited, t_step=time.time() - got)   # seconds waiting for batches / computing
                self.step += 1
                self.offset += len(batches)
                avg.add(logs)
                keys = TQDM_KEYS_A if phase_before == "A" else TQDM_KEYS_B
                mean = avg.mean()
                pbar.set_postfix({k: f"{mean[k]:.3f}" for k in keys if k in mean}, refresh=False)
                pbar.set_description(f"epoch {self.epoch} | phase {phase_before} | step {self.step}/{total}", refresh=False)
                boundary = self.step == cfg.phase_a_steps
                if self.step % cfg.log_every == 0 or boundary or self.step == total:
                    self.history["train"].append({"step": self.step, "epoch": self.epoch, "phase": phase_before,
                                                  "time": time.time() - t0, **avg.mean()})
                    avg.reset()
                if self.step % cfg.eval_every == 0 or boundary or self.step == total:
                    if boundary:  # evaluate the end of phase A with phase-A losses
                        self.step -= 1
                        last_val = self.validate()
                        self.step += 1
                    else:
                        last_val = self.validate()
                    self.history["valid"].append({"step": self.step, "epoch": self.epoch, "phase": phase_before, **last_val})
                    self._log_validation(last_val, phase_before)
                    self.plot_curves()
                    save_json(self.history, os.path.join(self.run_dir, "history.json"))
                    if phase_before == "B" and last_val.get("score", -1e9) > self.best["score"]:
                        self.best = {"score": last_val["score"], "step": self.step}
                        self.save_checkpoint("best.pt")
                    if phase_before == "A":
                        self._track_phase_a(last_val)
                if boundary:
                    self._restore_best_phase_a()
                    self.save_checkpoint("phase_a.pt")
                    self.log.info(f"[phase] representation phase finished at step {self.step}; codecs and grounder "
                                  f"are now frozen, training the writers and the reader")
                saved = self.step % cfg.save_every == 0 or boundary or self.step == total
                if saved:
                    self.save_checkpoint()
                if time_left() <= 0 and self.step < total:   # out of session time: keep this exact step, stop
                    pbar.close()
                    if not saved:
                        self.save_checkpoint()
                    save_json(self.history, os.path.join(self.run_dir, "history.json"))
                    raise OutOfTime(f"time budget used up at step {self.step}/{total} of '{cfg.exp_name}' "
                                    f"(phase {self.phase}); the step is saved, run again to continue from it")
                if boundary or self.step >= total:
                    break
                waited = time.time()
            pbar.close()
            if self.offset >= n_ep:
                self.epoch, self.offset = self.epoch + 1, 0
        if not os.path.exists(os.path.join(self.ckpt_dir, "best.pt")):
            self.save_checkpoint("best.pt")
        self.save_checkpoint()
        save_json(self.history, os.path.join(self.run_dir, "history.json"))
        self.plot_curves()
        self.log.info(f"[fit] done: {self.step} steps, " + (
            f"best score {self.best['score']:.3f} at step {self.best['step']}" if self.best["step"] >= 0
            else "representation phase only (no transport phase, hence no score)"))
        return last_val

    # ------------------------------------------------------------------ phase-A model selection
    def _phase_a_best_path(self) -> str:
        return os.path.join(self.ckpt_dir, "phase_a_best.pt")

    def _track_phase_a(self, val: Dict[str, float]) -> None:
        """Remember the phase-A state with the best validation hub losses.  Only states after the release are
        comparable (and usable: the null option and minimality exist from then on)."""
        cfg = self.cfg
        keys = ("ground", "unexplained", "read")
        if not cfg.phase_a_select or self.step < cfg.gate_release_frac * cfg.phase_a_steps or any(k not in val for k in keys):
            return
        value = float(sum(val[k] for k in keys))
        if value < self.best_a["value"]:
            self.best_a = {"value": value, "step": self.step}
            tmp = self._phase_a_best_path() + ".tmp"
            torch.save({"model": self.model.state_dict(), "step": self.step, "value": value}, tmp)
            os.replace(tmp, self._phase_a_best_path())

    def _restore_best_phase_a(self) -> None:
        """At the end of phase A: continue from the best validated state rather than from the last one."""
        if not self.cfg.phase_a_select or self.best_a["step"] in (-1, self.step) or not os.path.exists(self._phase_a_best_path()):
            return
        ck = torch.load(self._phase_a_best_path(), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.log.info(f"[phase] the transport phase starts from the phase-A state of step {ck['step']} "
                      f"(validation grounding + sufficiency + read = {ck['value']:.3f}, the best after the release); "
                      f"later states fit the training set better and the validation set worse")

    def _log_validation(self, val: Dict[str, float], phase: str) -> None:
        keys = [k for k in val if k in TQDM_KEYS_A + TQDM_KEYS_B or k.endswith(("/r1", "content_acc"))
                or k.startswith("tone/") or k == "score"]
        msg = " | ".join(f"{k} {val[k]:.3f}" for k in keys[:40])
        self.log.info(f"[valid] step {self.step} phase {phase} | {msg}")
        for k in [k for k in val if k.endswith("/cf1") and "/flat_" not in k and "/rt_" not in k]:
            d = k[:-len("/cf1")]   # one line per text direction: every score next to its chance level
            parts = [f"{lab} {val[f'{d}/{m}']:.1f} (chance {val[f'{d}/chance_{m}']:.1f})"
                     for m, lab in (("cf1", "content F1"), ("bleu", "BLEU"), ("chrf", "chrF++")) if f"{d}/chance_{m}" in val]
            parts += [f"{lab} {val[f'{d}/{m}']:.1f}" for m, lab in (("bleu_multi", "BLEU multi-ref"), ("wer", "WER")) if f"{d}/{m}" in val]
            self.log.info(f"[text] {d} | " + " | ".join(parts))
        recent = [h for h in self.history["train"][-max(1, self.cfg.eval_every // max(1, self.cfg.log_every)):] if "t_step" in h]
        if recent:
            data, step = np.mean([h["t_data"] for h in recent]), np.mean([h["t_step"] for h in recent])
            self.log.info(f"[speed] {1.0 / max(1e-9, data + step):.2f} steps/s | waiting for batches {data:.3f}s, "
                          f"computing {step:.3f}s per step" + (" (the data loader is the bottleneck)" if data > step else ""))
        if phase == "B":   # what the outputs actually look like: two examples per direction
            H = self.cfg.hub_langs[0]
            for name in [f"aud_{L}>txt_{H}" for L in self.cfg.island_langs] + [f"aud_{H}>txt_{H}", f"txt_{H}>txt_{H}"]:
                for s in self.evaluator.samples.get(name, [])[:2]:
                    self.log.info(f"[sample] {name} | ref: {s['ref'][:90]} | out: {s['hyp'][:90]}")

    # ------------------------------------------------------------------ curves
    def plot_curves(self) -> None:
        from .utils import setup_matplotlib

        setup_matplotlib()
        import matplotlib.pyplot as plt

        tr, va = self.history["train"], self.history["valid"]
        if not tr:
            return
        out = os.path.join(self.run_dir, "curves")
        pa = self.cfg.phase_a_steps

        def series(hist, key, phase=None):
            pts = [(h["step"], h[key]) for h in hist if key in h and (phase is None or h.get("phase") == phase)]
            return ([p[0] for p in pts], [p[1] for p in pts]) if pts else ([], [])

        panels = [("loss", "A", "phase A total loss"), ("txt_ce", "A", "text codec CE"), ("sem_mse", "A", "speech codec MSE"),
                  ("ground", "A", "grounding InfoNCE"), ("unexplained", "A", "latent variance NOT explained by the hub"),
                  ("read", "A", "reader error on the hub code (no image)"),
                  ("loss", "B", "phase B total loss"), ("cfm_txt", "B", "CFM text writer"), ("cfm_aud", "B", "CFM speech writer"),
                  ("cfm_img", "B", "CFM image writer"), ("read_aud", "B", "reader error, speech"),
                  ("bt_read", "B", "back-translation: read synthetic speech (train only)")]
        fig, axes = plt.subplots(3, 4, figsize=(19, 11))
        for ax, (key, ph, title) in zip(axes.ravel(), panels):
            xs, ys = series(tr, key, ph)
            xv, yv = series(va, key, ph)
            if xs:
                ax.plot(xs, ys, color="#1f77b4", lw=1.4, label="train")
            if xv:
                ax.plot(xv, yv, color="#d62728", lw=1.6, marker="o", ms=3.5, label="valid")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("step")
            ax.grid(alpha=0.25)
            if xs or xv:
                ax.legend(fontsize=8)
        fig.suptitle(f"{self.cfg.exp_name}: train vs valid losses (phase A = steps < {pa}, phase B after)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "losses_train_vs_valid.png"), dpi=130)
        plt.close(fig)

        mkeys = sorted({k for h in va for k in h if (k.endswith(("/cf1", "/chance_cf1", "/r1", "content_acc")) or k == "score")
                        and "/flat_" not in k and "/rt_" not in k})
        tone = sorted({k for h in tr for k in h if k.startswith("gate_")})
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        for k in mkeys[:16]:
            xs, ys = series(va, k)
            chance = k.endswith("/chance_cf1")
            axes[0].plot(xs, ys, marker=None if chance else "o", ms=3, ls=":" if chance else "-", lw=1.0 if chance else 1.5,
                         color="0.55" if chance else None, label=None if chance else k)
        axes[0].set_title("validation metrics through the hub (dotted grey: chance level of content F1)")
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=6.5, ncol=2)
        for k in tone:
            xs, ys = series(tr, k)
            axes[1].plot(xs, ys, label=k.replace("gate_", "gate "))
        axes[1].axvline(pa, color="k", ls=":", lw=1)
        axes[1].set_title("lexical-tonality gate per language (training)")
        axes[1].set_ylim(-0.02, 1.02)
        if tone:
            axes[1].legend(fontsize=8)
        for k, lab in (("bt_write", "write: real speech, pseudo code"), ("bt_read", "read: synthetic speech, real code"),
                       ("bt_shift", "transport shift of the pseudo codes"), ("slot_use_txt", "slot use (text)")):
            xs, ys = series(tr, k)
            if xs:
                axes[2].plot(xs, ys, label=lab)
        axes[2].set_title("hub-space back-translation and slot use")
        if axes[2].lines:
            axes[2].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "metrics_gates_backtranslation.png"), dpi=130)
        plt.close(fig)
