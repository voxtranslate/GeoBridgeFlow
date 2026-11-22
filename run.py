#!/usr/bin/env python
"""GeoBridge-Flow command line (see ``geobridge/cli.py`` for the stages).

    python run.py prepare --preset flickr8k_yo            # download + split + cache features (once)
    python run.py train   --preset flickr8k_yo            # train (auto-resumes from the last checkpoint)
    python run.py eval    --preset flickr8k_yo            # full test-set evaluation of best.pt
    python run.py infer   --preset flickr8k_yo            # figures, attributions, saved outputs
    python run.py ablate  --preset synth --groups hub island tone
    python run.py all     --preset synth                  # everything, on the controlled benchmark
"""
import sys

from geobridge.cli import main

if __name__ == "__main__":
    sys.exit(main())
