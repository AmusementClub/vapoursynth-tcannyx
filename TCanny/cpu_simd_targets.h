// GPL-3.0-or-later. Included first by every Highway source (cpu_simd.cpp and tests/direction_test.cpp), before any
// Highway header, so both compile the same targets.
#pragma once

// Only the targets `opt` can name (x86: SSE2, AVX2, AVX-512; aarch64: NEON), never the build's baseline. The names
// expand where Highway uses them.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS ((HWY_SSSE3 | HWY_SSE4 | HWY_AVX3_DL | HWY_AVX3_ZEN4 | HWY_AVX3_SPR | HWY_AVX10_2 | \
                               HWY_NEON_BF16 | HWY_ALL_SVE | HWY_EMU128 | HWY_SCALAR) & ~HWY_BASELINE_TARGETS)
#endif
