#!/bin/bash
# Reproducible build on the devhost (run from the repo root there). usage: build_devhost.sh <build-dir> [meson options...]
# Needs: Vulkan headers (VSVulkan4.h includes vulkan/vulkan_core.h), the R81 VapourSynth headers, ninja+meson.
# Defaults assume the nss-cuda-devhost layout; override VULKAN_HEADERS / VS_HEADERS elsewhere.
set -e
B=${1:?build dir}; shift
VK=${VULKAN_HEADERS:-$HOME/.cache/nss-deps/Vulkan-Headers-vulkan-sdk-1.4.363.0/include}
VS=${VS_HEADERS:-$HOME/src/vs-envs/r81/site/vapoursynth/include}
[ -f $B/build.ninja ] || meson setup $B -Dvulkan_headers=$VK -Dvapoursynth_headers=$VS --buildtype=release "$@"
ninja -C $B
