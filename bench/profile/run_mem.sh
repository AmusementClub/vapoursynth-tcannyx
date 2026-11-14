#!/bin/bash
cd "$(dirname "$0")"; export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty
P=${TC_PYTHON:-python}; PL=${PLUGIN:?set PLUGIN=path/to/libtcannyx.so}
for res in 1080 4k; do
 for spec in "cpu GRAYS natural 1" "cpu GRAYS natural 0" "vulkan GRAYS natural 1" "vulkan GRAYS natural 0" "vulkan GRAY8 natural 1" "vulkan GRAY8 natural 0" "cuda GRAYS natural 1" "cuda GRAYS natural 0" "vulkan GRAYS natural ''" "cuda GRAYS natural -1"; do
  set -- $spec; m=$4; [ "$m" = "''" ] && m=""
  for rep in 1 2; do
  $P mem.py $PL $1 $2 $3 "$m" 1.5 $res 8 200 2>&1 | tail -1
  done
 done
done
