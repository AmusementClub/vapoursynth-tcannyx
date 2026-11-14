#!/bin/bash
# CUDA counters and exact-depth runs (needs a -Dprofiling build). usage: PLUGIN=<profiling libtcannyx.so> TC_PYTHON=... run_instr_cuda.sh <label>
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
P=${TC_PYTHON:-python}; PL=${PLUGIN:?set PLUGIN}; O=results/instr_$1; mkdir -p $O /tmp/vs-empty
run() { tag=$1; shift; TCANNY_PROF=1 $P single.py $PL "$@" > $O/$tag.out 2> $O/$tag.err; }
for th in 1 4 16; do
 run cu_GRAYS_natural_m1_t$th cuda GRAYS natural 1 1.5 1 1080 $th 400
 run cu_GRAYS_natural_m0_t$th cuda GRAYS natural 0 1.5 1 1080 $th 400
done
run cu_GRAYS_natural_m-1_t1 cuda GRAYS natural -1 1.5 1 1080 1 400
for c in natural busy flat; do for res in 1080 4k; do n=60; [ $res = 4k ] && n=20
 TCANNY_PROF_BATCH=1 run depth_vk_GRAYS_${c}_${res} vulkan GRAYS $c 0 1.5 1 $res 4 $n
done; done
TCANNY_PROF_BATCH=1 run depth_vk_GRAYS_natural_s4_1080 vulkan GRAYS natural 0 4.0 1 1080 4 60
echo done > $O/done
