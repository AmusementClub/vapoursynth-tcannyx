#!/bin/bash
# usage: PLUGIN=<libtcannyx.so> TC_PYTHON=<r81 python> run_suite_named.sh <suite> <label>   -> results/<suite>_<label>.jsonl
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
mkdir -p /tmp/vs-empty results
${TC_PYTHON:-python} run_suite.py $1 results/$1_$2.jsonl > results/$1_$2.log 2>&1
echo ALLDONE >> results/$1_$2.log
