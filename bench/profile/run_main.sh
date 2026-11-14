#!/bin/bash
# sequential measurement run of the main suites (stock build)
cd "$(dirname "$0")";
export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty TC_THREADS=16
P=${TC_PYTHON:-python}
mkdir -p results
for s in baselines matrix_main resident matrix_ofat; do
  $P run_suite.py $s results/$s.jsonl --smi results/${s}_smi.csv >> results/$s.log 2>&1
done
echo ALLDONE >> results/main.done
