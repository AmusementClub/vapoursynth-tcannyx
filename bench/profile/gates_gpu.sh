#!/bin/bash
# GPU correctness gates of docs/perf-plan.md 1.1 on the devhost. usage: gates_gpu.sh <build dir with libtcannyx.so> [vulkan|cuda|both]
# Needs the R81 python (TC_PYTHON) and VulkanHeaders on CPATH only for building, not for running.
cd "$(dirname "$0")/../.."; B=${1:?build dir}; WHICH=${2:-both}
export VAPOURSYNTH_PLUGIN_PATH=/tmp/vs-empty; mkdir -p /tmp/vs-empty
PY=${TC_PYTHON:-python}; P=$B/libtcannyx.so; rc=0
ninja -C $B > /tmp/gates_build.log 2>&1 || { echo "BUILD FAILED (see /tmp/gates_build.log): refusing to test a stale binary"; exit 1; }
run() { echo "== $*"; "$@" 2>&1 | tail -${TAILN:-3} | cut -c1-400; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FAILED: $*"; rc=1; }; }
${MESON:-$PY -m mesonbuild.mesonmain} test -C $B --print-errorlogs 2>&1 | grep -E "^(Ok|Fail)|FAIL|ERROR"; [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FAILED: meson test"; rc=1; }
if [ $WHICH != cuda ]; then run $PY tests/vulkan_golden.py --plugin $P; run $PY tests/gpu_profile.py --backend vulkan --plugin $P; run $PY tests/vulkan_leak.py --plugin $P; fi
if [ $WHICH != vulkan ]; then run $PY tests/cuda_golden.py --plugin $P; run $PY tests/gpu_profile.py --backend cuda --plugin $P; run $PY tests/cuda_leak.py --plugin $P; run $PY tests/cuda_leak.py --plugin $P --threads 32 --frames 1500; run $PY tests/cuda_leak.py --plugin $P --format GRAY8 --frames 1500; run $PY tests/cuda_leak.py --plugin $P --format GRAY16 --frames 1500; fi
python3 tests/check_shader_twins.py || rc=1
for g in TCanny/shaders/*.comp.glsl; do glslangValidator -V -S comp $g -o /tmp/gate.spv >/dev/null || { echo "SHADER FAILED $g"; rc=1; }; done
# Drop-in check against a built upstream (tests/upstream_reference.sh): UPSTREAM=<its libtcanny.so>; UPSTREAM_OPTS="1 0 2 3"
# also compares the SIMD paths (needs the --fix-simd build); skipped when UPSTREAM is unset.
if [ -n "${UPSTREAM:-}" ]; then
  run $PY tests/upstream_compare.py --plugin $P --upstream $UPSTREAM --opt ${UPSTREAM_OPTS:-1}
  [ $WHICH != cuda ] && run $PY tests/upstream_compare.py --plugin $P --upstream $UPSTREAM --gpu vulkan
  [ $WHICH != vulkan ] && run $PY tests/upstream_compare.py --plugin $P --upstream $UPSTREAM --gpu cuda
else echo "SKIPPED upstream_compare (set UPSTREAM=<upstream r14 libtcanny.so>)"; fi
echo "gates rc=$rc"; exit $rc
