#!/bin/bash
# Launch E1 Brian2 module jobs in parallel, one process per module, pinned to cores.
# Usage: e1_launch.sh [POISSON_SEED]   (0 = training run; >0 = noise-band inference re-run)
# 18 jobs = 3 seeds x 6 modules (offsets 0,25 = calibration modules; 50..125 = test modules).
# Cores: 10 Cortex-X925 (5-9,15-19) + 8 Cortex-A725 (1-4,11-14); CPUs 0 and 10 left free.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PS=${1:-0}
CORES=(5 6 7 8 9 15 16 17 18 19 1 2 3 4 11 12 13 14)
mkdir -p $ROOT/logs/e1
n=0
for SEED in 0 10 20; do
  $ROOT/scripts/e1_setup_rundir.sh $SEED $PS > /dev/null
  for OFF in 0 25 50 75 100 125; do
    CORE=${CORES[$n]}
    nohup $ROOT/scripts/e1_job.sh $SEED $OFF $CORE $PS > /dev/null 2>&1 &
    echo "$! S$SEED O$OFF core$CORE P$PS" >> $ROOT/logs/e1/pids_P$PS.txt
    n=$((n+1))
  done
done
echo "launched $n jobs (P$PS)"
