#!/bin/bash
# Create e1/S$SEED (training run dir) or e1/S${SEED}_P$P (noise-band inference dir).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEED=$1; PS=${2:-0}
mkdir -p $ROOT/e1
[ -e $ROOT/e1/data ] || ln -s $ROOT/data $ROOT/e1/data          # ./../data from e1/<run>
if [ "$PS" = "0" ]; then RUN=$ROOT/e1/S$SEED; else RUN=$ROOT/e1/S${SEED}_P$PS; fi
mkdir -p $RUN
[ -e $RUN/dataset_imagenames ] || ln -s $ROOT/VPRSNN/dataset_imagenames $RUN/dataset_imagenames
if [ "$PS" != "0" ]; then
  # reuse trained weights + training-time outputs (assignment inputs) of the base run
  [ -e $RUN/weights ] || ln -s $ROOT/e1/S$SEED/weights $RUN/weights
  for d in $ROOT/e1/S$SEED/outputs/outputs_ne400_L25_offset*; do
    o=$RUN/outputs/$(basename $d); mkdir -p $o
    cp $d/resultPopVecs1500.npy $d/inputNumbers.npy $o/ 2>/dev/null
  done
fi
echo $RUN
