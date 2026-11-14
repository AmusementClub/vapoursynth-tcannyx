#!/bin/bash
cd "$(dirname "$0")"; export PATH=/usr/local/cuda/bin:$PATH VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
PL=${PLUGIN:?set PLUGIN=path/to/libtcannyx.so}; P=${TC_PYTHON:-python}
mkdir -p nsys; rm -f nsys/*
go() { tag=$1; shift; timeout 300 nsys profile --trace=cuda,vulkan,osrt --vulkan-gpu-workload=true -o nsys/$tag -f true --stats=false $P single.py $PL "$@" > nsys/$tag.out 2>&1; }
go vk_GRAYS_nat_m0_t1 vulkan GRAYS natural 0 1.5 1 1080 1 150
go vk_GRAYS_nat_m1_t1 vulkan GRAYS natural 1 1.5 1 1080 1 150
go vk_GRAYS_busy_m0_t1 vulkan GRAYS busy 0 1.5 1 1080 1 150
go vk_GRAY8_nat_m0_t1 vulkan GRAY8 natural 0 1.5 1 1080 1 150
go vk_GRAY8_nat_m1_t1 vulkan GRAY8 natural 1 1.5 1 1080 1 150
go vk_GRAYS_nat_m0_t16 vulkan GRAYS natural 0 1.5 1 1080 16 150
go vk_GRAYS_nat_m0_4k_t1 vulkan GRAYS natural 0 1.5 1 4k 1 40
go vk_updown_GRAYS_t1 vulkan GRAYS natural "" 1.5 1 1080 1 150
go cu_GRAYS_nat_m0_t1 cuda GRAYS natural 0 1.5 1 1080 1 150
go cu_GRAYS_nat_m1_t1 cuda GRAYS natural 1 1.5 1 1080 1 150
go cu_GRAYS_busy_m0_t1 cuda GRAYS busy 0 1.5 1 1080 1 150
go cu_GRAYS_nat_m0_t16 cuda GRAYS natural 0 1.5 1 1080 16 150
go cu_GRAYS_nat_m0_4k_t1 cuda GRAYS natural 0 1.5 1 4k 1 40
for f in nsys/*.nsys-rep; do b=${f%.nsys-rep}; nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,vulkan_api_sum --format csv -o $b --force-export=true $f > /dev/null 2>&1; done
echo done > nsys/nsys.done
