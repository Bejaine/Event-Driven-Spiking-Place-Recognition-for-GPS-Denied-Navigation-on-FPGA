#!/bin/bash
# Queue of extra E7 runs, executed after the l_hw run finishes (sequential, 12 procs each).
cd "$(dirname "$0")/.."
while pgrep -f "e7_run.py all l_hw 12" >/dev/null; do sleep 60; done
E7_PSEED=1 envs/vprsnn/bin/python scripts/e7_run.py all a_float 12 cal,test
E7_PSEED=1 envs/vprsnn/bin/python scripts/e7_run.py all l_hw 12 cal,test
envs/vprsnn/bin/python scripts/e7_run.py all l_hw_dt05 12 cal,test
envs/vprsnn/bin/python scripts/e7_run.py all f_nodelay 12 cal,test
