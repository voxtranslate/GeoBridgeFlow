"""Pipeline stages and the command line.

    prepare   download the sources, write the split JSON, build the one-time feature cache
    train     two-phase training (resumes from the last checkpoint by itself)
    eval      metrics of best.pt on a split
    infer     figures, internal views of the model, saved outputs
    ablate    the ablation suite and its tables

    python run.py prepare --preset flickr8k_yo
    python run.py train   --preset flickr8k_yo
    python run.py eval    --preset flickr8k_yo
    python run.py infer   --preset flickr8k_yo
    python run.py ablate  --preset synth --groups hub island tone
    python run.py all     --preset synth

Any config field can be overridden:  --set bridge_frac=0.05 hub_mode=global seed=1
"""
from __future__ import annotations

import argparse
import gc
import json
from typing import Any, Dict, Optional, Sequence

import torch

from .ablation import AblationRunner
from .config import Config
from .data import GeoBridgeData
from .inferencer import Inferencer
from .trainer import Trainer
from .utils import OutOfTime, get_logger, resolve_device, set_time_budget

STAGES = ("prepare", "train", "eval", "infer", "ablate")


def build_config(preset: str, overrides: Optional[Dict[str, Any]] = None, exp_name: str = "") -> Config:
    cfg = Config.preset(preset).update(dict(overrides or {}))
    if exp_name:
        cfg.exp_name = exp_name
    elif cfg.exp_name == "geobridge":
        cfg.exp_name = preset
    return cfg


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_stages(cfg: Config, stages: Sequence[str], split: str = "test", checkpoint: str = "", resume_from: str = "",
               groups: Optional[Sequence[str]] = None, only: Optional[Sequence[str]] = None,
               seeds: Sequence[int] = (0,), collect: bool = False) -> Dict[str, Any]:
    """Run the requested stages in pipeline order.  Every stage is restartable: finished work (downloads,
    caches, checkpoints, ablation cells) is found on disk and skipped."""
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {STAGES}")
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    out: Dict[str, Any] = {}
    data: Optional[GeoBridgeData] = None
    if "prepare" in stages or "train" in stages:
        data = GeoBridgeData(cfg, get_logger(), resolve_device(cfg.device))
        out["data"] = data.describe()
    if "train" in stages:
        trainer = Trainer(cfg, data)
        if resume_from:
            trainer.resume_from(resume_from)
        out["valid"] = trainer.fit()
        del trainer
        release_memory()
    if "eval" in stages or "infer" in stages:
        inf = Inferencer(cfg, checkpoint=checkpoint or None, data=data)
        if "eval" in stages:
            out[split] = inf.evaluate(split)
        if "infer" in stages:
            out["report"] = inf.run_all(split)
        del inf
        release_memory()
    if "ablate" in stages:
        runner = AblationRunner(cfg, seeds=seeds)
        out["ablation"] = runner.collect(groups) if collect else runner.run(groups, only)
    return out


def main(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[*STAGES, "all"])
    ap.add_argument("--preset", default="flickr8k_yo")
    ap.add_argument("--exp_name", default="")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume_from", default="", help="explicit checkpoint (any step) instead of last.pt")
    ap.add_argument("--checkpoint", default="", help="checkpoint for eval / infer (default: best.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", nargs="*", default=[], help="ablation groups (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--list", action="store_true", help="list ablation groups and exit")
    ap.add_argument("--only", nargs="*", default=[], help="run only variants whose name contains one of these")
    ap.add_argument("--collect", action="store_true", help="rebuild ablation tables from finished cells only")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="stop training cleanly after this many hours (the step is saved; run again to continue)")
    args = ap.parse_args(argv)
    set_time_budget(args.hours)
    cfg = build_config(args.preset, dict(kv.split("=", 1) for kv in args.set), args.exp_name)
    if args.command == "ablate" and args.list:
        AblationRunner(cfg, seeds=args.seeds).describe()
        return
    stages = STAGES if args.command == "all" else (args.command,)
    try:
        out = run_stages(cfg, stages, split=args.split, checkpoint=args.checkpoint, resume_from=args.resume_from,
                         groups=args.groups or None, only=args.only or None, seeds=args.seeds, collect=args.collect)
    except OutOfTime as e:
        get_logger().info(f"[time] {e}")
        return 75   # EX_TEMPFAIL: not finished, safe to run again
    if args.command == "prepare":
        print(json.dumps(out["data"], indent=2, default=str))
    if args.command == "eval":
        print(json.dumps({k: round(v, 3) for k, v in out[args.split].items()}, indent=1))
