#!/usr/bin/env bash
# Reference ablation on the CPU-sized SynthWorld preset: one seed, 1,500 flow steps per cell.
# Usage: scripts/cpu_ablation_queue.sh <stream-id> "<group>:<variant substring>" ...
# Every invocation also runs the full model of the group (once; it is shared), so train that first with
#   scripts/cpu_ablation_queue.sh 0 "private:__full_only__"
# before starting several streams.  Finished cells are skipped, so the script can be re-run or split.
set -u
PY=${PYTHON:-python}
STREAM=$1; shift
COMMON="--preset synth_fast --exp_name sf --set phase_b_steps=1500 eval_every=750 save_every=750 milestone_every=1500 eval_max_items=80 torch_threads=${THREADS:-2}"
for item in "$@"; do
  group="${item%%:*}"; only="${item#*:}"
  echo "[stream $STREAM] $(date +%H:%M:%S) start $group / $only"
  $PY run.py ablate $COMMON --groups "$group" --only "$only" > "/tmp/abl_${STREAM}_${group}.log" 2>&1
  echo "[stream $STREAM] $(date +%H:%M:%S) done  $group / $only (exit $?)"
done
echo "[stream $STREAM] ALL DONE"
