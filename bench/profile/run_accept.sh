#!/bin/bash
# Acceptance set (docs/perf-plan.md section 4). usage: PLUGIN=<libtcannyx.so> TC_PYTHON=<r81 python> run_accept.sh <label>
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
mkdir -p /tmp/vs-empty results
${TC_PYTHON:-python} run_suite.py acceptance results/accept_$1.jsonl --smi results/accept_$1_smi.csv > results/accept_$1.log 2>&1
echo ALLDONE >> results/accept_$1.log
