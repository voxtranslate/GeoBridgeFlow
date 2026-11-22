#!/usr/bin/env bash
# The complete study for the paper (needs a GPU; every step resumes if interrupted).
# Budget on one 24 GB GPU, rough: main run ~1 day; each ablation cell 4-10 h; SynthWorld cells ~1 h.
set -euo pipefail
PY=${PYTHON:-python}
SEEDS="0 1 2"

# 1. real benchmark: download, split, cache frozen features (once)
$PY run.py prepare --preset flickr8k_yo

# 2. main table: bridge budgets, three seeds
for s in $SEEDS; do
  for b in 1.0 0.2 0.05 0.01; do
    $PY run.py train --preset flickr8k_yo --exp_name "main_b${b}_s${s}" --set seed=$s bridge_frac=$b
    $PY run.py eval  --preset flickr8k_yo --exp_name "main_b${b}_s${s}" --set seed=$s bridge_frac=$b
  done
done

# 3. strict zero-text anchor (DINOv2 has never seen text; SigLIP has)
for s in $SEEDS; do
  $PY run.py train --preset flickr8k_yo --exp_name "dinov2_s${s}" --set seed=$s vision_encoder=facebook/dinov2-base
  $PY run.py eval  --preset flickr8k_yo --exp_name "dinov2_s${s}" --set seed=$s vision_encoder=facebook/dinov2-base
done

# 4. ablations on real data (the three claims) and on the controlled benchmark (everything)
$PY run.py ablate --preset flickr8k_yo --exp_name real --groups hub read island bridge tone protocol --seeds $SEEDS
$PY run.py ablate --preset synth       --exp_name synth --seeds $SEEDS

# 5. figures and saved outputs of the reference run
$PY run.py infer --preset flickr8k_yo --exp_name main_b1.0_s0 --set seed=0 bridge_frac=1.0
