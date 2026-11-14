#!/bin/bash
# usage: run_phase.sh <label> <plugin> [instrumented plugin]   (sequential: counters, full acceptance, upstream CPU)
cd "$(dirname "$0")"; export TC_PYTHON=${TC_PYTHON:-python}
[ -n "$3" ] && PLUGIN=$3 ./run_instr_vk.sh $1
PLUGIN=$2 ./run_accept.sh $1
VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty $TC_PYTHON run_suite.py acceptance_upstream results/acceptance_upstream_r12.jsonl --plugin "$UPSTREAM" --plugin-force > results/acceptance_upstream_r12.log 2>&1
echo PHASEDONE > results/phase_$1.done
