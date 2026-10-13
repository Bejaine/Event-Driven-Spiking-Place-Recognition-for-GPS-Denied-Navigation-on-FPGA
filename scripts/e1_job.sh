#!/bin/bash
# One E1 module job: train(+record+calibrate) then test, via the repo pipeline.
# Usage: e1_job.sh SEED OFFSET CORE [POISSON_SEED]
#   POISSON_SEED=0 (default): full train/record/calibrate/test in e1/S$SEED
#   POISSON_SEED>0          : inference-only re-run (record/calibrate/test) in e1/S${SEED}_P$P,
#                             reusing the trained weights of e1/S$SEED (Brian2 noise band)
# Resumable: a module is skipped when its .done marker exists.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEED=$1; OFF=$2; CORE=$3; PS=${4:-0}
PY=$ROOT/envs/vprsnn/bin/python
NCAL=50
if [ "$PS" = "0" ]; then RUN=$ROOT/e1/S$SEED; else RUN=$ROOT/e1/S${SEED}_P$PS; fi
mkdir -p $RUN/markers $RUN/timing $ROOT/logs/e1
MARK=$RUN/markers/O$OFF.done
[ -f $MARK ] && { echo "skip $RUN O$OFF"; exit 0; }
export TMPDIR=$ROOT/cache/b2_S${SEED}_P${PS}_O$OFF; mkdir -p $TMPDIR
export WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $RUN
LOG=$ROOT/logs/e1/S${SEED}_P${PS}_O$OFF.log
run() { taskset -c $CORE $PY $ROOT/scripts/e1_module.py --seed $SEED --offset $OFF --poisson_seed $PS "$@" >> $LOG 2>&1; }
echo "start $(date -Iseconds) core=$CORE" >> $LOG
if [ "$PS" = "0" ]; then
  run --mode train --timing_out timing/O${OFF}_train.json || exit 1
else
  run --mode record --timing_out timing/O${OFF}_record.json || exit 1
fi
if [ $OFF -ge $NCAL ]; then
  run --mode test --timing_out timing/O${OFF}_test.json || exit 1
fi
echo "end $(date -Iseconds)" >> $LOG
touch $MARK
