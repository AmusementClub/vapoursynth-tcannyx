#!/bin/bash
# Builds the upstream TCanny used as the drop-in reference by tests/upstream_compare.py.
# usage: upstream_reference.sh <work dir> <dir containing VapourSynth4.h> [--fix-simd]
#   Prints the path of the built libtcanny.so on the last line.
# Pinned to upstream commit 14ac2ce (the last commit after the r14 tag; the r14 tag itself fails to run, see below).
# --fix-simd turns the SIMD sources' `auto& x{ Vec4f().load_a(...) }` (a reference to a temporary, undefined behaviour that
# GCC 15 compiles into all-zero output) into `auto x{ ... }`, which is the only change; without it only `opt=1` (scalar) is usable.
set -euo pipefail
W=${1:?work dir}; INC=${2:?vapoursynth include dir}; FIX=${3:-}
mkdir -p "$W"; cd "$W"
[ -d src ] || git clone -q https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny.git src
cd src; git checkout -q 14ac2ce
if [ "$FIX" = "--fix-simd" ]; then
python3 - <<'P'
import glob, re
n = 0
for f in glob.glob("TCanny/TCanny_*.cpp"):
    s = open(f).read()
    t, c = re.subn(r"auto& *(\w+)\{ *(Vec\w+\(\)\.load\w*\()", r"auto \1{ \2", s)
    n += c
    open(f, "w").write(t)
print("patched", n, "references")
P
fi
mkdir -p ../pc; cat > ../pc/vapoursynth.pc <<P
prefix=$W
libdir=$W/lib
includedir=$INC
Name: vapoursynth
Description: headers only
Version: 81
Cflags: -I\${includedir}
P
rm -rf build; PKG_CONFIG_PATH="$W/pc" meson setup build >/dev/null; ninja -C build >/dev/null
echo "$W/src/build/libtcanny.so"
