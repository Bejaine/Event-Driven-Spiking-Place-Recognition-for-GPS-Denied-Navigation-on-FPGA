#!/bin/bash
# Orin run queue (sequential, resumable: a step is skipped when its output exists).
#   1. J4 CUDA port      -> results/orin_cuda.json
#   2. J4 torch engine   -> results/orin_gpu.json
#   3. J3 Brian2 timing  -> results/orin_brian2_timing.json   (pinned to core 3)
#   4. J6 consistency    -> results/orin_e4_l_hw.json, results/orin_e3_summary.json
# J6 runs in cache/j6root (symlinks to code and e1/S{0,10,20}) because e4_activity.py,
# e3_run.py and e3_evaluate.py write fixed paths (results/e4_*, e3/, results/e3_summary.json)
# that hold the DGX results in this repo.
# Usage: nohup scripts/orin_queue.sh > logs/orin/queue.log 2>&1 &
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/envs/vprsnn/bin/python
export TMPDIR=$ROOT/cache/tmp WANDB_MODE=disabled
mkdir -p "$TMPDIR"
cd "$ROOT"
step() { echo "=== $(date -Iseconds) $*"; }

if [ ! -s results/orin_cuda.json ]; then
  step "J4 CUDA port"; $PY scripts/orin_gpu.py cuda
fi
if [ ! -s results/orin_gpu.json ]; then
  step "J4 torch engine"; $PY scripts/orin_gpu.py torch
fi
if [ ! -s results/orin_brian2_timing.json ]; then
  step "J3 Brian2"
  (cd e1/S0_P9 && taskset -c 3 $PY ../../scripts/orin_brian2_timing.py run 0 50 "$ROOT/logs/orin/j3_brian2_raw.json") \
    > logs/orin/j3_brian2.log 2>&1
  $PY scripts/orin_brian2_timing.py summary logs/orin/j3_brian2_raw.json > results/orin_brian2_timing.json
  cat results/orin_brian2_timing.json
fi

J6=$ROOT/cache/j6root
mkdir -p $J6/e1 $J6/results
for d in snnsim scripts golden_c VPRSNN; do [ -e $J6/$d ] || ln -s $ROOT/$d $J6/$d; done
for s in 0 10 20; do [ -e $J6/e1/S$s ] || ln -s $ROOT/e1/S$s $J6/e1/S$s; done
if [ ! -s results/orin_e4_l_hw.json ]; then
  step "J6 e4_activity l_hw seed 0"
  $PY $J6/scripts/e4_activity.py l_hw --seeds 0 > logs/orin/j6_e4.log 2>&1 && cp $J6/results/e4_l_hw.json results/orin_e4_l_hw.json
  tail -20 logs/orin/j6_e4.log
fi
if [ ! -s results/orin_e3_summary.json ]; then
  step "J6 e3_run l_hw P0"; $PY $J6/scripts/e3_run.py l_hw 0 --seeds 0 10 20 > logs/orin/j6_e3_l_hw.log 2>&1
  step "J6 e3_run a_float P0"; $PY $J6/scripts/e3_run.py a_float 0 --seeds 0 10 20 > logs/orin/j6_e3_a_float.log 2>&1
  step "J6 e3_evaluate"; (cd $J6 && $PY $J6/scripts/e3_evaluate.py) > logs/orin/j6_e3_evaluate.log 2>&1 \
    && cp $J6/results/e3_summary.json results/orin_e3_summary.json
  tail -5 logs/orin/j6_e3_evaluate.log
fi
step "gzip power logs"; gzip -9 -f logs/orin/power/*.csv 2>/dev/null
step "QUEUE DONE"
