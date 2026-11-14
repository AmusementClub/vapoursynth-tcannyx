/*
**   VapourSynth port by HolyWu
**
**                 tcanny v1.0 for Avisynth 2.5.x
**
**   Copyright (C) 2009 Kevin Stone
**
**   This program is free software: you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation, either version 3 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <cmath>
#include <cstddef>
#include <mutex>
#include <new>
#include <stdexcept>

#include <string>

#include "cpu_reference.h"

using namespace std::literals;

template<typename pixel_t>
static void gaussianBlur(const pixel_t* _srcp, float* VS_RESTRICT temp, float* VS_RESTRICT dstp, const int width, const int height,
                         const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radiusH, const int radiusV,
                         const float* weightsH, const float* weightsV) noexcept {
    auto diameter{ radiusV * 2 + 1 };
    auto srcp{ std::make_unique<const pixel_t* []>(diameter) };

    srcp[radiusV] = _srcp;
    for (auto i{ 1 }; i <= radiusV; i++)
        srcp[radiusV - i] = srcp[radiusV + i] = srcp[radiusV] + srcStride * i;

    weightsH += radiusH;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ 0 }; v < diameter; v++)
                sum += srcp[v][x] * weightsV[v];

            temp[x] = sum;
        }

        for (auto i{ 1 }; i <= radiusH; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ -radiusH }; v <= radiusH; v++)
                sum += temp[x + v] * weightsH[v];

            dstp[x] = sum;
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        srcp[diameter - 1] += (y < height - 1 - radiusV) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurH(const pixel_t* srcp, float* VS_RESTRICT temp, float* VS_RESTRICT dstp, const int width, const int height,
                          const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radius, const float* weights) noexcept {
    weights += radius;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++)
            temp[x] = srcp[x];

        for (auto i{ 1 }; i <= radius; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ -radius }; v <= radius; v++)
                sum += temp[x + v] * weights[v];

            dstp[x] = sum;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurV(const pixel_t* _srcp, float* VS_RESTRICT dstp, const int width, const int height,
                          const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radius, const float* weights) noexcept {
    auto diameter{ radius * 2 + 1 };
    auto srcp{ std::make_unique<const pixel_t* []>(diameter) };

    srcp[radius] = _srcp;
    for (auto i{ 1 }; i <= radius; i++)
        srcp[radius - i] = srcp[radius + i] = srcp[radius] + srcStride * i;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto sum{ 0.0f };

            for (auto v{ 0 }; v < diameter; v++)
                sum += srcp[v][x] * weights[v];

            dstp[x] = sum;
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            srcp[i] = srcp[i + 1];
        srcp[diameter - 1] += (y < height - 1 - radius) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void copyPlane(const pixel_t* srcp, float* VS_RESTRICT dstp, const int width, const int height,
                      const ptrdiff_t srcStride, const ptrdiff_t dstStride) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++)
            dstp[x] = srcp[x];

        srcp += srcStride;
        dstp += dstStride;
    }
}

static void detectEdge(float* VS_RESTRICT blur, float* VS_RESTRICT gradient, int* VS_RESTRICT direction, const int width, const int height,
                       const ptrdiff_t stride, const ptrdiff_t bgStride, const int mode, const int op, const float scale) noexcept {
    auto cur{ blur };
    auto next{ blur + bgStride };
    auto next2{ blur + bgStride * 2 };
    auto prev{ next };
    auto prev2{ next2 };

    cur[-1] = cur[1];
    cur[width] = cur[width - 2];
    if (op == FDOG) {
        cur[-2] = cur[2];
        cur[width + 1] = cur[width - 3];
    }

    for (auto y{ 0 }; y < height; y++) {
        next[-1] = next[1];
        next[width] = next[width - 2];
        if (op == FDOG) {
            next[-2] = next[2];
            next[width + 1] = next[width - 3];

            next2[-1] = next2[1];
            next2[-2] = next2[2];
            next2[width] = next2[width - 2];
            next2[width + 1] = next2[width - 3];
        }

        for (auto x{ 0 }; x < width; x++) {
            float gx{}, gy{};

            if (op != FDOG) {
                auto c1{ prev[x - 1] };
                auto c2{ prev[x] };
                auto c3{ prev[x + 1] };
                auto c4{ cur[x - 1] };
                auto c6{ cur[x + 1] };
                auto c7{ next[x - 1] };
                auto c8{ next[x] };
                auto c9{ next[x + 1] };

                switch (op) {
                case TRITICAL:
                    gx = c6 - c4;
                    gy = c2 - c8;
                    break;
                case PREWITT:
                    gx = (c3 + c6 + c9 - c1 - c4 - c7) / 2.0f;
                    gy = (c1 + c2 + c3 - c7 - c8 - c9) / 2.0f;
                    break;
                case SOBEL:
                    gx = c3 + 2.0f * c6 + c9 - c1 - 2.0f * c4 - c7;
                    gy = c1 + 2.0f * c2 + c3 - c7 - 2.0f * c8 - c9;
                    break;
                case SCHARR:
                    gx = 3.0f * c3 + 10.0f * c6 + 3.0f * c9 - 3.0f * c1 - 10.0f * c4 - 3.0f * c7;
                    gy = 3.0f * c1 + 10.0f * c2 + 3.0f * c3 - 3.0f * c7 - 10.0f * c8 - 3.0f * c9;
                    break;
                case KROON:
                    gx = 17.0f * c3 + 61.0f * c6 + 17.0f * c9 - 17.0f * c1 - 61.0f * c4 - 17.0f * c7;
                    gy = 17.0f * c1 + 61.0f * c2 + 17.0f * c3 - 17.0f * c7 - 61.0f * c8 - 17.0f * c9;
                    break;
                case KIRSCH:
                    auto g1{ 5.0f * c1 + 5.0f * c2 + 5.0f * c3 - 3.0f * c4 - 3.0f * c6 - 3.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g2{ 5.0f * c1 + 5.0f * c2 - 3.0f * c3 + 5.0f * c4 - 3.0f * c6 - 3.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g3{ 5.0f * c1 - 3.0f * c2 - 3.0f * c3 + 5.0f * c4 - 3.0f * c6 + 5.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g4{ -3.0f * c1 - 3.0f * c2 - 3.0f * c3 + 5.0f * c4 - 3.0f * c6 + 5.0f * c7 + 5.0f * c8 - 3.0f * c9 };
                    auto g5{ -3.0f * c1 - 3.0f * c2 - 3.0f * c3 - 3.0f * c4 - 3.0f * c6 + 5.0f * c7 + 5.0f * c8 + 5.0f * c9 };
                    auto g6{ -3.0f * c1 - 3.0f * c2 - 3.0f * c3 - 3.0f * c4 + 5.0f * c6 - 3.0f * c7 + 5.0f * c8 + 5.0f * c9 };
                    auto g7{ -3.0f * c1 - 3.0f * c2 + 5.0f * c3 - 3.0f * c4 + 5.0f * c6 - 3.0f * c7 - 3.0f * c8 + 5.0f * c9 };
                    auto g8{ -3.0f * c1 + 5.0f * c2 + 5.0f * c3 - 3.0f * c4 + 5.0f * c6 - 3.0f * c7 - 3.0f * c8 - 3.0f * c9 };
                    auto g{ std::max({ std::abs(g1), std::abs(g2), std::abs(g3), std::abs(g4), std::abs(g5), std::abs(g6), std::abs(g7), std::abs(g8) }) };
                    gradient[x] = g * scale;
                    break;
                }
            } else {
                auto c1{ prev2[x - 2] };
                auto c2{ prev2[x - 1] };
                auto c3{ prev2[x] };
                auto c4{ prev2[x + 1] };
                auto c5{ prev2[x + 2] };
                auto c6{ prev[x - 2] };
                auto c7{ prev[x - 1] };
                auto c8{ prev[x] };
                auto c9{ prev[x + 1] };
                auto c10{ prev[x + 2] };
                auto c11{ cur[x - 2] };
                auto c12{ cur[x - 1] };
                auto c14{ cur[x + 1] };
                auto c15{ cur[x + 2] };
                auto c16{ next[x - 2] };
                auto c17{ next[x - 1] };
                auto c18{ next[x] };
                auto c19{ next[x + 1] };
                auto c20{ next[x + 2] };
                auto c21{ next2[x - 2] };
                auto c22{ next2[x - 1] };
                auto c23{ next2[x] };
                auto c24{ next2[x + 1] };
                auto c25{ next2[x + 2] };

                gx = c5 + 2.0f * c10 + 3.0f * c15 + 2.0f * c20 + c25 + c4 + 2.0f * c9 + 3.0f * c14 + 2.0f * c19 + c24
                    - c2 - 2.0f * c7 - 3.0f * c12 - 2.0f * c17 - c22 - c1 - 2.0f * c6 - 3.0f * c11 - 2.0f * c16 - c21;
                gy = c1 + 2.0f * c2 + 3.0f * c3 + 2.0f * c4 + c5 + c6 + 2.0f * c7 + 3.0f * c8 + 2.0f * c9 + c10
                    - c16 - 2.0f * c17 - 3.0f * c18 - 2.0f * c19 - c20 - c21 - 2.0f * c22 - 3.0f * c23 - 2.0f * c24 - c25;
            }

            if (op != KIRSCH) {
                gx *= scale;
                gy *= scale;
                gradient[x] = std::sqrt(gx * gx + gy * gy);
            }

            if (mode == 0) {
                auto dr{ std::atan2(gy, gx) };
                if (dr < 0.0f)
                    dr += M_PIF;

                auto bin{ static_cast<int>(dr * 4.0f * M_1_PIF + 0.5f) };
                direction[x] = (bin >= 4) ? 0 : bin;
            }
        }

        prev2 = prev;
        prev = cur;
        cur = next;
        if (op != FDOG) {
            next += (y < height - 2) ? bgStride : -bgStride;
        } else {
            next = next2;
            next2 += (y < height - 3) ? bgStride : -bgStride;
        }
        gradient += bgStride;
        direction += stride;
    }
}

static void nonMaximumSuppression(const int* direction, float* VS_RESTRICT gradient, float* VS_RESTRICT blur, const int width, const int height,
                                  const ptrdiff_t stride, const ptrdiff_t bgStride, const int radiusAlign) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        auto row{ gradient + bgStride * y };
        row[-1] = row[1];
        row[width] = row[width - 2];
    }
    const ptrdiff_t offsets[]{ 1, -bgStride + 1, -bgStride, -bgStride - 1 };

    gradient[-1] = gradient[1];
    gradient[-1 + bgStride * (height - 1)] = gradient[1 + bgStride * (height - 1)];
    gradient[width] = gradient[width - 2];
    gradient[width + bgStride * (height - 1)] = gradient[width - 2 + bgStride * (height - 1)];
    std::copy_n(gradient - radiusAlign + bgStride, width + radiusAlign * 2, gradient - radiusAlign - bgStride);
    std::copy_n(gradient - radiusAlign + bgStride * (height - 2), width + radiusAlign * 2, gradient - radiusAlign + bgStride * height);

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            auto offset{ offsets[direction[x]] };
            blur[x] = (gradient[x] >= std::max(gradient[x + offset], gradient[x - offset])) ? gradient[x] : fltLowest;
        }

        direction += stride;
        gradient += bgStride;
        blur += bgStride;
    }
}

template<typename pixel_t>
static void binarizeCE(const float* srcp, pixel_t* VS_RESTRICT dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                       const int peak) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            if constexpr (std::is_integral_v<pixel_t>)
                dstp[x] = (srcp[x] == fltMax) ? static_cast<pixel_t>(peak) : 0;
            else
                dstp[x] = (srcp[x] == fltMax) ? 1.0f : 0.0f;
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t, bool clampFP = true>
static void discretizeGM(const float* srcp, pixel_t* VS_RESTRICT dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                         const int peak) noexcept {
    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x++) {
            if constexpr (std::is_integral_v<pixel_t>)
                dstp[x] = static_cast<pixel_t>(std::min(static_cast<int>(srcp[x] + 0.5f), peak));
            else if constexpr (clampFP)
                dstp[x] = std::clamp(srcp[x], 0.0f, 1.0f);
            else
                dstp[x] = srcp[x];
        }

        srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void filter_c(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, TCannyScratch* scratch, const VSAPI* vsapi) noexcept {
    for (auto plane{ 0 }; plane < d->vi->format.numPlanes; plane++) {
        if (d->process[plane]) {
            const auto width{ vsapi->getFrameWidth(src, plane) };
            const auto height{ vsapi->getFrameHeight(src, plane) };
            const auto stride{ vsapi->getStride(src, plane) / d->vi->format.bytesPerSample };
            const auto bgStride{ stride + d->radiusAlign * 2 };
            auto srcp{ reinterpret_cast<const pixel_t*>(vsapi->getReadPtr(src, plane)) };
            auto dstp{ reinterpret_cast<pixel_t*>(vsapi->getWritePtr(dst, plane)) };

            auto blur{ scratch->blur.get() + d->radiusAlign };
            auto gradient{ scratch->gradient.get() + bgStride + d->radiusAlign };
            auto direction{ scratch->direction.get() };
            auto found{ scratch->found.get() };

            if (d->radiusH[plane] && d->radiusV[plane])
                gaussianBlur(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->radiusV[plane],
                             d->weightsH[plane].data(), d->weightsV[plane].data());
            else if (d->radiusH[plane])
                gaussianBlurH(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->weightsH[plane].data());
            else if (d->radiusV[plane])
                gaussianBlurV(srcp, blur, width, height, stride, bgStride, d->radiusV[plane], d->weightsV[plane].data());
            else
                copyPlane(srcp, blur, width, height, stride, bgStride);

            if (d->mode != -1) {
                detectEdge(blur, gradient, direction, width, height, stride, bgStride, d->mode, d->op, d->scale);

                if (d->mode == 0) {
                    nonMaximumSuppression(direction, gradient, blur, width, height, stride, bgStride, d->radiusAlign);
                    hysteresis(blur, found, scratch->coordinates, width, height, bgStride, d->t_h, d->t_l);
                }
            }

            if (d->mode == 0)
                binarizeCE(blur, dstp, width, height, bgStride, stride, d->peak);
            else if (d->mode == 1)
                discretizeGM(gradient, dstp, width, height, bgStride, stride, d->peak);
            else
                discretizeGM<pixel_t, false>(blur, dstp, width, height, bgStride, stride, d->peak);
        }
    }
}


namespace {
// Free list of scratch objects. A frame takes one for its whole duration and puts it back on every exit path, so the
// list grows to the number of frames that run at once and no frame allocates or zero-fills 25+ MB on its own.
struct TCannyCpuPool {
    std::mutex mutex;
    std::vector<std::unique_ptr<TCannyScratch>> free;
};

class ScratchLease {
public:
    explicit ScratchLease(TCannyCpuPool &pool) : pool_(pool) {
        {
            std::lock_guard<std::mutex> lock(pool_.mutex);
            if (!pool_.free.empty()) {
                scratch_ = std::move(pool_.free.back());
                pool_.free.pop_back();
            }
        }
        if (!scratch_) scratch_ = std::make_unique<TCannyScratch>();
    }
    ~ScratchLease() {
        try {
            std::lock_guard<std::mutex> lock(pool_.mutex);
            pool_.free.push_back(std::move(scratch_));
        } catch (...) {
            // Out of memory while returning the object: drop it, the next frame allocates a new one.
        }
    }
    ScratchLease(const ScratchLease &) = delete;
    ScratchLease &operator=(const ScratchLease &) = delete;
    TCannyScratch *get() const noexcept { return scratch_.get(); }

private:
    TCannyCpuPool &pool_;
    std::unique_ptr<TCannyScratch> scratch_;
};

template<typename T, typename Ptr>
void ensureBuffer(Ptr &buffer, size_t &capacity, size_t count, size_t alignment) {
    if (buffer && capacity >= count) return;
    buffer.reset(vsh::vsh_aligned_malloc<T>(count * sizeof(T), alignment));
    if (!buffer) { capacity = 0; throw std::bad_alloc(); }
    capacity = count;
}
} // namespace

void tcannySetupCpu(TCannyData *d) {
    const int bytesPerSample = d->vi->format.bytesPerSample;
    d->filter = d->opt == 1 ? nullptr : tcannySelectSimd(bytesPerSample, d->opt);
    if (!d->filter)
        d->filter = bytesPerSample == 1 ? filter_c<uint8_t> : bytesPerSample == 2 ? filter_c<uint16_t> : filter_c<float>;
    d->alignment = 64;
    const int radius = std::max({d->radiusH[0], d->radiusH[1], d->radiusH[2], d->op == FDOG ? 2 : 1});
    d->radiusAlign = (radius + 15) & ~15;
    d->cpuPool = new TCannyCpuPool();
}

void tcannyDestroyCpuPool(void *pool) noexcept {
    delete static_cast<TCannyCpuPool *>(pool);
}

void tcannyProcessCpu(const VSFrame *src, VSFrame *dst, const TCannyData *d, const VSAPI *vsapi) {
    // Size the scratch buffers for the largest processed plane rather than assuming plane 0 is.
    size_t stride = 0, height = 0, width = 0;
    for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
        if (!d->process[plane]) continue;
        stride = std::max(stride, static_cast<size_t>(vsapi->getStride(src, plane) / d->vi->format.bytesPerSample));
        height = std::max(height, static_cast<size_t>(vsapi->getFrameHeight(src, plane)));
        width = std::max(width, static_cast<size_t>(vsapi->getFrameWidth(src, plane)));
    }
    const size_t pitch = stride + d->radiusAlign * 2;
    ScratchLease lease(*static_cast<TCannyCpuPool *>(d->cpuPool));
    TCannyScratch &scratch = *lease.get();
    ensureBuffer<float>(scratch.blur, scratch.blurCount, pitch * height, d->alignment);
    ensureBuffer<float>(scratch.gradient, scratch.gradientCount, pitch * (height + 2), d->alignment);
    // Even non-edge modes advance this pointer in the original SIMD loop.
    ensureBuffer<int>(scratch.direction, scratch.directionCount, stride * height, d->alignment);
    const int radiusV = std::max({d->radiusV[0], d->radiusV[1], d->radiusV[2]});
    if (d->opt != 1 && d->vi->format.bytesPerSample < 4 && radiusV > 0)
        ensureBuffer<float>(scratch.blurRows, scratch.blurRowsCount, tcannyBlurRowsPitch(static_cast<int>(width)) * (radiusV * 2 + 1), d->alignment);
    if (d->mode == 0 && d->opt != 1)
        ensureBuffer<uint8_t>(scratch.edgeMap, scratch.edgeMapCount, tcannyEdgeMapStride(static_cast<int>(width)) * (height + 2), d->alignment);
    if (!scratch.found || scratch.foundCount < width * height) {
        scratch.found.reset(new bool[width * height]);
        scratch.foundCount = width * height;
    }
    // The gradient used to be zeroed for every frame. Output does not depend on the halo (TCANNY_POISON_SCRATCH=2
    // poisons it without changing any result), so it is only cleared when the layout changes, to keep reads defined.
    if (scratch.gradientPitch != pitch || scratch.gradientRows != height) {
        std::fill_n(scratch.gradient.get(), scratch.gradientCount, 0.0f);
        scratch.gradientPitch = pitch;
        scratch.gradientRows = height;
    }
#ifdef TCANNY_POISON_SCRATCH
    // Test builds: make any read of stale scratch contents visible. Level 2 also poisons the gradient halo.
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::fill_n(scratch.blur.get(), pitch * height, nan);
        std::fill_n(scratch.direction.get(), stride * height, 0x7FFFFFFF);
        std::fill_n(scratch.found.get(), width * height, true);
        if (scratch.edgeMap) std::fill_n(scratch.edgeMap.get(), scratch.edgeMapCount, uint8_t{0xFF});
        if (scratch.blurRows) std::fill_n(scratch.blurRows.get(), scratch.blurRowsCount, nan);
        for (size_t y = 0; y < height; y++)
            std::fill_n(scratch.gradient.get() + (y + 1) * pitch + d->radiusAlign, stride, nan);
#if TCANNY_POISON_SCRATCH >= 2
        std::fill_n(scratch.gradient.get(), pitch, nan);
        for (size_t y = 0; y < height; y++) {
            float *row = scratch.gradient.get() + (y + 1) * pitch;
            std::fill_n(row, d->radiusAlign, nan);
            std::fill_n(row + d->radiusAlign + stride, d->radiusAlign, nan);
        }
        std::fill_n(scratch.gradient.get() + (height + 1) * pitch, pitch, nan);
#endif
    }
#endif
    d->filter(src, dst, d, &scratch, vsapi);
}
