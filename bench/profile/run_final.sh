#!/bin/bash
# Final snapshot: full acceptance set and the F1 transfer-size sweep. usage: PLUGIN=<libtcannyx.so> TC_PYTHON=... run_final.sh <label>
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
mkdir -p results /tmp/vs-empty
./run_accept.sh $1
${TC_PYTHON:-python} f1_updown.py "$PLUGIN" > results/f1_updown_$1.jsonl 2>&1
echo FINALDONE > results/final_$1.done
