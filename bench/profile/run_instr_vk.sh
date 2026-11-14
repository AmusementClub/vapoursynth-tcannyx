#!/bin/bash
# Vulkan mode-0 hysteresis counters (needs an instrumented build). usage: PLUGIN=<instrumented libtcannyx.so> TC_PYTHON=... run_instr_vk.sh <outdir-label>
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
P=${TC_PYTHON:-python}; PL=${PLUGIN:?set PLUGIN}; O=results/instr_$1; mkdir -p $O /tmp/vs-empty
run() { tag=$1; shift; TCANNY_PROF=1 $P single.py $PL "$@" > $O/$tag.out 2> $O/$tag.err; }
for th in 1 16; do
 for c in natural busy flat; do run vk_GRAYS_${c}_m0_t$th vulkan GRAYS $c 0 1.5 1 1080 $th 400; done
 run vk_GRAY8_natural_m0_t$th vulkan GRAY8 natural 0 1.5 1 1080 $th 400
 run vk_GRAYS_natural_m0_s4_t$th vulkan GRAYS natural 0 4.0 1 1080 $th 400
 run vk_GRAYS_natural_m1_t$th vulkan GRAYS natural 1 1.5 1 1080 $th 400
done
run vk_GRAYS_natural_m0_4k_t16 vulkan GRAYS natural 0 1.5 1 4k 16 150
for c in natural busy flat; do TCANNY_PROF_BATCH=1 run depth_vk_GRAYS_${c}_1080 vulkan GRAYS $c 0 1.5 1 1080 4 40; done
echo done > $O/done
