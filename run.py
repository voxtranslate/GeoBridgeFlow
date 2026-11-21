#!/usr/bin/env python
"""GeoBridge-Flow command line.

    python run.py prepare --preset flickr8k_yo            # download + split + cache features (once)
    python run.py train   --preset flickr8k_yo            # train (auto-resumes from the last checkpoint)
    python run.py eval    --preset flickr8k_yo            # full test-set evaluation of best.pt
    python run.py infer   --preset flickr8k_yo            # figures, attributions, saved outputs
    python run.py ablate  --preset flickr8k_yo --groups anchor docking tone
    python run.py all     --preset synth                  # everything, on the controlled benchmark

Any config field can be overridden:  --set bridge_frac=0.01 dock_mode=gw seed=1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from geobridge.config import Config


def build_config(args) -> Config:
    cfg = Config.preset(args.preset)
    over = {}
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        over[k] = v
    cfg.update(over)
    if args.exp_name:
        cfg.exp_name = args.exp_name
    elif cfg.exp_name == "geobridge":
        cfg.exp_name = args.preset
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "train", "eval", "infer", "ablate", "all"])
    ap.add_argument("--preset", default="flickr8k_yo")
    ap.add_argument("--exp_name", default="")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume_from", default="", help="explicit checkpoint (any step) instead of last.pt")
    ap.add_argument("--checkpoint", default="", help="checkpoint for eval / infer (default: best.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--groups", nargs="*", default=[], help="ablation groups (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--list", action="store_true", help="list ablation groups and exit")
    args = ap.parse_args()
    cfg = build_config(args)
    if cfg.torch_threads:
        import torch

        torch.set_num_threads(cfg.torch_threads)

    if args.command == "prepare":
        from geobridge.data import GeoBridgeData
        from geobridge.utils import get_logger, resolve_device

        data = GeoBridgeData(cfg, get_logger(), resolve_device(cfg.device))
        print(json.dumps(data.describe(), indent=2, default=str))
        return

    if args.command in ("train", "all"):
        from geobridge.trainer import Trainer

        tr = Trainer(cfg)
        if args.resume_from:
            tr.resume_from(args.resume_from)
        tr.fit()
        if args.command == "train":
            return

    if args.command in ("eval", "infer", "all"):
        from geobridge.inferencer import Inferencer

        inf = Inferencer(cfg, checkpoint=args.checkpoint or None)
        if args.command in ("eval", "all"):
            m = inf.evaluate(args.split)
            print(json.dumps({k: round(v, 3) for k, v in m.items()}, indent=1))
        if args.command in ("infer", "all"):
            inf.run_all(args.split)
        if args.command != "all":
            return

    if args.command in ("ablate", "all"):
        from geobridge.ablation import AblationRunner

        runner = AblationRunner(cfg, seeds=args.seeds)
        if args.list:
            runner.describe()
            return
        runner.run(args.groups or None)


if __name__ == "__main__":
    sys.exit(main())
