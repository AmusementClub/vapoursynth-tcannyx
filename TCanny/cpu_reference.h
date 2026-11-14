// GPL-3.0-or-later; adapted from HomeOfVapourSynthEvolution/VapourSynth-TCanny.
#pragma once
#include "TCanny.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
static constexpr float M_PIF = 3.14159265358979323846f;
static constexpr float M_1_PIF = 0.318309886183790671538f;
static constexpr float fltMax = std::numeric_limits<float>::max();
static constexpr float fltLowest = std::numeric_limits<float>::lowest();

using TCannyFilter = void (*)(const VSFrame *, VSFrame *, const TCannyData *, TCannyScratch *, const VSAPI *) noexcept;
// SIMD filter for `opt` (cpu_simd.cpp): 0 = best target this CPU has (nullptr if none), 2/3/4 = SSE2/AVX2/AVX-512.
// Throws if the CPU or architecture lacks the requested one.
TCannyFilter tcannySelectSimd(int bytesPerSample, int opt);

using unique_float = std::unique_ptr<float[], decltype(&vsh::vsh_aligned_free)>;
using unique_int = std::unique_ptr<int[], decltype(&vsh::vsh_aligned_free)>;
using unique_u8 = std::unique_ptr<uint8_t[], decltype(&vsh::vsh_aligned_free)>;

// Row pitch of the SIMD path's mode-0 edge map (cpu_simd.cpp): a one-pixel zero border on each side plus room for a
// whole vector written past the last pixel, at least 64 so that a 64-byte read from any interior row stays in the map.
inline size_t tcannyEdgeMapStride(int width) noexcept { return (static_cast<size_t>(width) + 18 + 63) & ~static_cast<size_t>(63); }
// Row pitch of the SIMD path's ring of converted source rows (vertical blur of integer formats): whole 16-float vectors.
inline size_t tcannyBlurRowsPitch(int width) noexcept { return (static_cast<size_t>(width) + 15) & ~static_cast<size_t>(15); }


// Per-frame working memory. Instances are pooled (see tcannyProcessCpu) and reused across frames, so nothing here may
// be assumed to hold zeros or the previous frame's values; the one exception is the gradient halo (see gradientPitch).
struct TCannyScratch {
    unique_float blur{nullptr, vsh::vsh_aligned_free};
    unique_float gradient{nullptr, vsh::vsh_aligned_free};
    unique_int direction{nullptr, vsh::vsh_aligned_free};
    std::unique_ptr<bool[]> found;
    std::vector<std::pair<int, int>> coordinates;
    unique_u8 edgeMap{nullptr, vsh::vsh_aligned_free};  // SIMD mode 0: (height + 2) rows of tcannyEdgeMapStride(width)
    std::vector<int> edgeStack;
    size_t blurCount{}, gradientCount{}, directionCount{}, foundCount{}, edgeMapCount{};
    size_t gradientPitch{};
    size_t gradientRows{};
    // Last on purpose: inserting these between the members above made GRAY8 mode -1 8% slower on Apple M4 (NEON, which
    // does not even use them), reproducibly; a placement effect, not code.
    unique_float blurRows{nullptr, vsh::vsh_aligned_free};  // SIMD vertical blur of integer formats: diameter rows of tcannyBlurRowsPitch
    size_t blurRowsCount{};
};
// Upstream's hysteresis, kept as the scalar reference (opt=1); the SIMD path computes the same set with an edge map.
static inline void hysteresis(float* VS_RESTRICT srcp, bool* VS_RESTRICT found, std::vector<std::pair<int, int>>& coordinates, const int width,
                       const int height, const ptrdiff_t stride, const float t_h, const float t_l) noexcept {
    std::fill_n(found, static_cast<size_t>(width) * height, false);
    coordinates.clear();

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            if (!found[width * y + x] && srcp[stride * y + x] >= t_h) {
                srcp[stride * y + x] = fltMax;
                found[width * y + x] = true;

                coordinates.emplace_back(std::make_pair(x, y));

                while (!coordinates.empty()) {
                    const auto pos{ coordinates.back() };
                    coordinates.pop_back();

                    const auto xxStart{ std::max(pos.first - 1, 0) };
                    const auto xxStop{ std::min(pos.first + 1, width - 1) };
                    const auto yyStart{ std::max(pos.second - 1, 0) };
                    const auto yyStop{ std::min(pos.second + 1, height - 1) };

                    for (auto yy{ yyStart }; yy <= yyStop; yy++) {
                        for (auto xx{ xxStart }; xx <= xxStop; xx++) {
                            if (!found[width * yy + xx] && srcp[stride * yy + xx] >= t_l) {
                                srcp[stride * yy + xx] = fltMax;
                                found[width * yy + xx] = true;

                                coordinates.emplace_back(std::make_pair(xx, yy));
                            }
                        }
                    }
                }
            }
        }
    }
}
