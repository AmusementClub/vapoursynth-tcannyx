#!/bin/bash
# instrumented-build counter runs: one graph per process; stderr holds the TCANNY_PROF dump
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
P=${TC_PYTHON:-python}; PL=${PLUGIN:?set PLUGIN=path/to/instrumented libtcannyx.so}
mkdir -p results/instr; O=results/instr
run() { # tag be fmt content modes sigma op res threads frames opt [env]
  tag=$1; shift
  TCANNY_PROF=1 $P single.py $PL "$@" > $O/$tag.out 2> $O/$tag.err
}
for th in 1 16; do
 for c in natural busy flat; do
  run vk_GRAYS_${c}_m0_t$th vulkan GRAYS $c 0 1.5 1 1080 $th 400
  run cu_GRAYS_${c}_m0_t$th cuda GRAYS $c 0 1.5 1 1080 $th 400
  run vk_GRAY8_${c}_m0_t$th vulkan GRAY8 $c 0 1.5 1 1080 $th 400
 done
 for m in 1 -1; do
  run vk_GRAYS_natural_m${m}_t$th vulkan GRAYS natural $m 1.5 1 1080 $th 400
  run cu_GRAYS_natural_m${m}_t$th cuda GRAYS natural $m 1.5 1 1080 $th 400
  run vk_GRAY8_natural_m${m}_t$th vulkan GRAY8 natural $m 1.5 1 1080 $th 400
 done
done
for c in natural busy; do
 run vk_GRAYS_${c}_m0_4k_t16 vulkan GRAYS $c 0 1.5 1 4k 16 150
 run cu_GRAYS_${c}_m0_4k_t16 cuda GRAYS $c 0 1.5 1 4k 16 150
 run vk_GRAY8_${c}_m0_4k_t16 vulkan GRAY8 $c 0 1.5 1 4k 16 150
done
# YUV444PS / 3-plane
run vk_YUV444PS_natural_m0_t16 vulkan YUV444PS natural 0 1.5 1 1080 16 300
run cu_YUV444PS_natural_m0_t16 cuda YUV444PS natural 0 1.5 1 1080 16 300
# exact hysteresis depth: batch forced to 1 pass (counting only; timings meaningless)
for c in natural busy flat; do
 for res in 1080 4k; do
  n=40; [ $res = 4k ] && n=12
  TCANNY_PROF_BATCH=1 run depth_vk_GRAYS_${c}_${res} vulkan GRAYS $c 0 1.5 1 $res 4 $n
  TCANNY_PROF_BATCH=1 run depth_cu_GRAYS_${c}_${res} cuda GRAYS $c 0 1.5 1 $res 4 $n
  TCANNY_PROF_BATCH=1 run depth_vk_GRAY8_${c}_${res} vulkan GRAY8 $c 0 1.5 1 $res 4 $n
 done
done
# CPU stage timers (thread-time shares), threads 16 and 1
for th in 16 1; do
 for fmt in GRAYS GRAY8; do
  for opt in 1 3; do
   for m in -1 1 0; do
    for sg in 1.5 0; do
     run cpu_${fmt}_opt${opt}_m${m}_s${sg}_t$th cpu $fmt natural $m $sg 1 1080 $th 300 $opt
    done
   done
  done
 done
done
echo done > $O/instr.done
