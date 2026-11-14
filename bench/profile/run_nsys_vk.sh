#!/bin/bash
# nsys traces of Vulkan mode 0 (per-dispatch GPU times). usage: PLUGIN=<libtcannyx.so> TC_PYTHON=... run_nsys_vk.sh <label>
cd "$(dirname "$0")"; export PATH=/usr/local/cuda/bin:$PATH VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
P=${TC_PYTHON:-python}; PL=${PLUGIN:?set PLUGIN}; O=results/nsys_$1; mkdir -p $O /tmp/vs-empty
go() { tag=$1; shift; timeout 300 nsys profile --trace=vulkan,osrt --vulkan-gpu-workload=true -o $O/$tag -f true --stats=false $P single.py $PL "$@" > $O/$tag.out 2>&1
       nsys export -t sqlite -o $O/$tag.sqlite --force-overwrite=true $O/$tag.nsys-rep > /dev/null 2>&1; }
go vk_GRAYS_nat_m0_t1 vulkan GRAYS natural 0 1.5 1 1080 1 150
go vk_GRAYS_nat_m1_t1 vulkan GRAYS natural 1 1.5 1 1080 1 150
go vk_GRAYS_busy_m0_t1 vulkan GRAYS busy 0 1.5 1 1080 1 150
go vk_GRAY8_nat_m0_t1 vulkan GRAY8 natural 0 1.5 1 1080 1 150
go vk_GRAYS_nat_m0_4k_t1 vulkan GRAYS natural 0 1.5 1 4k 1 40
go vk_GRAYS_nat_m0_s4_t1 vulkan GRAYS natural 0 4.0 1 1080 1 150
echo done > $O/done
