#!/bin/bash
# F2: same-core comparison against upstream TCanny r12 (API 3, so it needs the R78 core).
# usage: R78_PYTHON=<python of the R78 venv> NOAUTO=<dir with nss_noauto> OURS=<cpu libtcannyx.so> UPSTREAM=<libtcanny-r12.so> run_f2.sh
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty PYTHONPATH=${NOAUTO:?}
mkdir -p results /tmp/vs-empty
$R78_PYTHON run_suite.py acceptance_upstream results/acceptance_upstream_r12.jsonl --plugin "$UPSTREAM" --plugin-force > results/acceptance_upstream_r12.log 2>&1
$R78_PYTHON run_suite.py acceptance_cpu results/acceptance_cpu_r78_ours.jsonl --plugin "$OURS" > results/acceptance_cpu_r78_ours.log 2>&1
echo F2DONE > results/f2.done
