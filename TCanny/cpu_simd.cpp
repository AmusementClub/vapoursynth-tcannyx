// GPL-3.0-or-later; adapted from HomeOfVapourSynthEvolution/VapourSynth-TCanny.
//
// CPU SIMD path: one Highway source compiled for every target (SSE2, AVX2 and AVX-512 on x86, NEON on aarch64).
// It is a port, operation for operation, of upstream r14's VCL kernels (its TCanny_SSE2/AVX2/AVX512.cpp), so x86 output stays
// bit-identical to upstream at opt 2/3/4: keep the expression trees (and MulAdd/MulSub where VCL had mul_add/mul_sub)
// exactly as they are. Dispatch is per instance through tcannySelectSimd, not Highway's process-wide chosen target,
// because `opt` is a filter argument.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu_simd.cpp"
#include "cpu_simd_targets.h"
#include <hwy/foreach_target.h>  // re-includes this file once per target
#include <hwy/highway.h>

#include <stdexcept>

#include "cpu_direction-inl.h"  // DF/DI/VF, k, ifAdd, atan2Vcl, directionBin

#include "cpu_reference.h"

HWY_BEFORE_NAMESPACE();
namespace tcanny_simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template<typename pixel_t>
HWY_INLINE VF loadPixels(const pixel_t* p) {
    const DF df;
    if constexpr (std::is_same_v<pixel_t, float>) {
        return hn::LoadU(df, p);
    } else {
        const hn::Rebind<pixel_t, DF> dp;
        return hn::ConvertTo(df, hn::PromoteTo(DI(), hn::LoadU(dp, p)));
    }
}

// Stores into the scratch planes, which the next stage reads back. Upstream uses non-temporal stores; the choice does
// not change any value. Measured 2026-11-15: cached stores are 2-10% faster at 1 thread (C4 and 5950X) but 3-28%
// slower at 16 threads on the 5950X, where bandwidth decides, so non-temporal stays the default.
#ifndef TCANNY_NT_SCRATCH
#define TCANNY_NT_SCRATCH 1
#endif
template<class D>
HWY_INLINE void storeScratch(hn::VFromD<D> v, D d, hn::TFromD<D>* p) {
#if TCANNY_NT_SCRATCH
    hn::Stream(v, d, p);
#else
    hn::Store(v, d, p);
#endif
}

// Converting integer rows once (below) pays where converting per tap costs more than reading 4x the bytes. Measured
// 2026-11-15, 8-bit, 1 thread: Zen 3 SSE2 -14%, AVX2 -4% (16 threads: within noise); Emerald Rapids AVX2 -2..-6%, but
// AVX-512 +1..+10% and Apple M4 NEON +8..+10%. So SSE2 and AVX2 only; TCANNY_BLUR_RING=0/1 overrides.
#ifndef TCANNY_BLUR_RING
#define TCANNY_BLUR_RING (HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_AVX2)
#endif
template<typename pixel_t>
constexpr bool kBlurRing{ TCANNY_BLUR_RING && !std::is_same_v<pixel_t, float> };

// Upstream's vertical blur window: the source row of every tap for the current output row, reflected at the top and the
// bottom (upstream walks row pointers; these are the same rows as indices). Integer rows are converted to float once,
// into a ring keyed by row index, instead of once per tap: a window never holds more than `diameter` distinct rows,
// all within a contiguous range of that length, so slots do not collide. The conversion is exact, so every sum is
// unchanged. Used only when kBlurRing; otherwise the blurs walk row pointers exactly as upstream.
template<typename pixel_t>
class VerticalWindow {
public:
    VerticalWindow(const pixel_t* src, ptrdiff_t srcStride, int width, int height, int radius, float* ring, size_t ringPitch)
        : src_{ src }, srcStride_{ srcStride }, width_{ width }, height_{ height }, radius_{ radius }, diameter_{ radius * 2 + 1 },
          rows_{ std::make_unique<int[]>(diameter_) }, held_{ std::make_unique<int[]>(diameter_) },
          taps_{ std::make_unique<const float*[]>(diameter_) }, ring_{ ring }, ringPitch_{ ringPitch } {
        rows_[radius] = 0;
        for (auto i{ 1 }; i <= radius; i++)
            rows_[radius - i] = rows_[radius + i] = i;
        std::fill_n(held_.get(), diameter_, -1);
    }

    // Rows of the taps for the current output row (converting the ones not in the ring yet).
    const float* const* taps() noexcept {
        for (auto v{ 0 }; v < diameter_; v++) {
            const auto r{ rows_[v] };
            const auto slot{ r % diameter_ };
            float* row{ ring_ + ringPitch_ * slot };
            if (held_[slot] != r) {
                const DF df;
                const pixel_t* s{ src_ + srcStride_ * r };
                for (auto x{ 0 }; x < width_; x += static_cast<int>(hn::Lanes(df)))
                    hn::Store(loadPixels(s + x), df, row + x);
                held_[slot] = r;
            }
            taps_[v] = row;
        }
        return taps_.get();
    }

    void advance(int y) noexcept {  // after output row y, as upstream's pointer walk
        for (auto i{ 0 }; i < diameter_ - 1; i++)
            rows_[i] = rows_[i + 1];
        rows_[diameter_ - 1] += (y < height_ - 1 - radius_) ? 1 : -1;
    }

private:
    const pixel_t* src_;
    ptrdiff_t srcStride_;
    int width_, height_, radius_, diameter_;
    std::unique_ptr<int[]> rows_, held_;
    std::unique_ptr<const float*[]> taps_;
    float* ring_;
    size_t ringPitch_;
};

template<typename pixel_t>
static void gaussianBlur(const pixel_t* __srcp, float* temp, float* dstp, const int width, const int height,
                         const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radiusH, const int radiusV,
                         const float* weightsH, const float* weightsV) noexcept {
    const DF df;
    const int step{ static_cast<int>(hn::Lanes(df)) };
    auto diameter{ radiusV * 2 + 1 };
    auto _srcp{ std::make_unique<const pixel_t* []>(diameter) };

    _srcp[radiusV] = __srcp;
    for (auto i{ 1 }; i <= radiusV; i++)
        _srcp[radiusV - i] = _srcp[radiusV + i] = _srcp[radiusV] + srcStride * i;

    weightsH += radiusH;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += step) {
            auto sum{ hn::Zero(df) };

            for (auto v{ 0 }; v < diameter; v++)
                sum = hn::MulAdd(loadPixels(_srcp[v] + x), hn::Set(df, weightsV[v]), sum);

            hn::Store(sum, df, temp + x);
        }

        for (auto i{ 1 }; i <= radiusH; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x += step) {
            auto sum{ hn::Zero(df) };

            for (auto v{ -radiusH }; v <= radiusH; v++)
                sum = hn::MulAdd(hn::LoadU(df, temp + x + v), hn::Set(df, weightsH[v]), sum);

            storeScratch(sum, df, dstp + x);
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            _srcp[i] = _srcp[i + 1];
        _srcp[diameter - 1] += (y < height - 1 - radiusV) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

// Vertical + horizontal blur of an integer plane through the converted-row ring (kBlurRing); same sums as gaussianBlur.
template<typename pixel_t>
static void gaussianBlurRing(const pixel_t* srcp, float* temp, float* dstp, const int width, const int height,
                             const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radiusH, const int radiusV,
                             const float* weightsH, const float* weightsV, float* ring, const size_t ringPitch) noexcept {
    const DF df;
    const int step{ static_cast<int>(hn::Lanes(df)) };
    const auto diameter{ radiusV * 2 + 1 };
    VerticalWindow<pixel_t> window{ srcp, srcStride, width, height, radiusV, ring, ringPitch };
    weightsH += radiusH;

    for (auto y{ 0 }; y < height; y++) {
        const float* const* taps{ window.taps() };
        for (auto x{ 0 }; x < width; x += step) {
            auto sum{ hn::Zero(df) };

            for (auto v{ 0 }; v < diameter; v++)
                sum = hn::MulAdd(hn::LoadU(df, taps[v] + x), hn::Set(df, weightsV[v]), sum);

            hn::Store(sum, df, temp + x);
        }

        for (auto i{ 1 }; i <= radiusH; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x += step) {
            auto sum{ hn::Zero(df) };

            for (auto v{ -radiusH }; v <= radiusH; v++)
                sum = hn::MulAdd(hn::LoadU(df, temp + x + v), hn::Set(df, weightsH[v]), sum);

            storeScratch(sum, df, dstp + x);
        }

        window.advance(y);
        dstp += dstStride;
    }
}

// Vertical-only blur of an integer plane through the converted-row ring (kBlurRing); same sums as gaussianBlurV.
template<typename pixel_t>
static void gaussianBlurVRing(const pixel_t* srcp, float* dstp, const int width, const int height, const ptrdiff_t srcStride,
                              const ptrdiff_t dstStride, const int radius, const float* weights, float* ring, const size_t ringPitch) noexcept {
    const DF df;
    const int step{ static_cast<int>(hn::Lanes(df)) };
    const auto diameter{ radius * 2 + 1 };
    VerticalWindow<pixel_t> window{ srcp, srcStride, width, height, radius, ring, ringPitch };

    for (auto y{ 0 }; y < height; y++) {
        const float* const* taps{ window.taps() };
        for (auto x{ 0 }; x < width; x += step) {
            auto sum{ hn::Zero(df) };

            for (auto v{ 0 }; v < diameter; v++)
                sum = hn::MulAdd(hn::LoadU(df, taps[v] + x), hn::Set(df, weights[v]), sum);

            storeScratch(sum, df, dstp + x);
        }
        window.advance(y);
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurH(const pixel_t* _srcp, float* temp, float* dstp, const int width, const int height,
                          const ptrdiff_t srcStride, const ptrdiff_t dstStride, const int radius, const float* weights) noexcept {
    const DF df;
    const int step{ static_cast<int>(hn::Lanes(df)) };
    weights += radius;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += step)
            hn::Store(loadPixels(_srcp + x), df, temp + x);

        for (auto i{ 1 }; i <= radius; i++) {
            temp[-i] = temp[i];
            temp[width - 1 + i] = temp[width - 1 - i];
        }

        for (auto x{ 0 }; x < width; x += step) {
            auto sum{ hn::Zero(df) };

            for (auto v{ -radius }; v <= radius; v++)
                sum = hn::MulAdd(hn::LoadU(df, temp + x + v), hn::Set(df, weights[v]), sum);

            storeScratch(sum, df, dstp + x);
        }

        _srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void gaussianBlurV(const pixel_t* __srcp, float* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                          const int radius, const float* weights) noexcept {
    const DF df;
    const int step{ static_cast<int>(hn::Lanes(df)) };
    auto diameter{ radius * 2 + 1 };
    auto _srcp{ std::make_unique<const pixel_t* []>(diameter) };

    _srcp[radius] = __srcp;
    for (auto i{ 1 }; i <= radius; i++)
        _srcp[radius - i] = _srcp[radius + i] = _srcp[radius] + srcStride * i;

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += step) {
            auto sum{ hn::Zero(df) };

            for (auto v{ 0 }; v < diameter; v++)
                sum = hn::MulAdd(loadPixels(_srcp[v] + x), hn::Set(df, weights[v]), sum);

            storeScratch(sum, df, dstp + x);
        }

        for (auto i{ 0 }; i < diameter - 1; i++)
            _srcp[i] = _srcp[i + 1];
        _srcp[diameter - 1] += (y < height - 1 - radius) ? srcStride : -srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
static void copyPlane(const pixel_t* srcp, float* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride) noexcept {
    const DF df;
    const int step{ static_cast<int>(hn::Lanes(df)) };

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += step)
            storeScratch(loadPixels(srcp + x), df, dstp + x);

        srcp += srcStride;
        dstp += dstStride;
    }
}

// Ring rows are read back at once: plain stores. Planes go through storeScratch (non-temporal, as upstream).
template<bool kRing, class D>
HWY_INLINE void storeRow(hn::VFromD<D> v, D d, hn::TFromD<D>* p) {
    if constexpr (kRing)
        hn::Store(v, d, p);
    else
        storeScratch(v, d, p);
}

// One row of upstream's detectEdge: gradient magnitude, and the direction bin when `direction` is not null (mode 0).
template<bool kRing>
HWY_INLINE void edgeRow(const float* prev2, const float* prev, const float* cur, const float* next, const float* next2,
                        float* gradient, int* direction, const int width, const int op, const float scale) {
    const DF df;
    const DI di;
    const int step{ static_cast<int>(hn::Lanes(df)) };
    for (auto x{ 0 }; x < width; x += step) {
        VF gx{ hn::Zero(df) }, gy{ hn::Zero(df) };

        if (op != FDOG) {
            auto c1{ hn::LoadU(df, prev + x - 1) };
            auto c2{ hn::LoadU(df, prev + x) };
            auto c3{ hn::LoadU(df, prev + x + 1) };
            auto c4{ hn::LoadU(df, cur + x - 1) };
            auto c6{ hn::LoadU(df, cur + x + 1) };
            auto c7{ hn::LoadU(df, next + x - 1) };
            auto c8{ hn::LoadU(df, next + x) };
            auto c9{ hn::LoadU(df, next + x + 1) };

            switch (op) {
            case TRITICAL:
                gx = c6 - c4;
                gy = c2 - c8;
                break;
            case PREWITT:
                gx = (c3 + c6 + c9 - c1 - c4 - c7) * k(0.5f);
                gy = (c1 + c2 + c3 - c7 - c8 - c9) * k(0.5f);
                break;
            case SOBEL:
                gx = c3 + hn::MulAdd(k(2.0f), c6, c9) - c1 - hn::MulAdd(k(2.0f), c4, c7);
                gy = c1 + hn::MulAdd(k(2.0f), c2, c3) - c7 - hn::MulAdd(k(2.0f), c8, c9);
                break;
            case SCHARR:
                gx = hn::MulAdd(k(3.0f), c3 + c9, k(10.0f) * c6) - hn::MulAdd(k(3.0f), c1 + c7, k(10.0f) * c4);
                gy = hn::MulAdd(k(3.0f), c1 + c3, k(10.0f) * c2) - hn::MulAdd(k(3.0f), c7 + c9, k(10.0f) * c8);
                break;
            case KROON:
                gx = hn::MulAdd(k(17.0f), c3 + c9, k(61.0f) * c6) - hn::MulAdd(k(17.0f), c1 + c7, k(61.0f) * c4);
                gy = hn::MulAdd(k(17.0f), c1 + c3, k(61.0f) * c2) - hn::MulAdd(k(17.0f), c7 + c9, k(61.0f) * c8);
                break;
            case KIRSCH:
                auto g1{ hn::MulSub(k(5.0f), c1 + c2 + c3, k(3.0f) * (c4 + c6 + c7 + c8 + c9)) };
                auto g2{ hn::MulSub(k(5.0f), c1 + c2 + c4, k(3.0f) * (c3 + c6 + c7 + c8 + c9)) };
                auto g3{ hn::MulSub(k(5.0f), c1 + c4 + c7, k(3.0f) * (c2 + c3 + c6 + c8 + c9)) };
                auto g4{ hn::MulSub(k(5.0f), c4 + c7 + c8, k(3.0f) * (c1 + c2 + c3 + c6 + c9)) };
                auto g5{ hn::MulSub(k(5.0f), c7 + c8 + c9, k(3.0f) * (c1 + c2 + c3 + c4 + c6)) };
                auto g6{ hn::MulSub(k(5.0f), c6 + c8 + c9, k(3.0f) * (c1 + c2 + c3 + c4 + c7)) };
                auto g7{ hn::MulSub(k(5.0f), c3 + c6 + c9, k(3.0f) * (c1 + c2 + c4 + c7 + c8)) };
                auto g8{ hn::MulSub(k(5.0f), c2 + c3 + c6, k(3.0f) * (c1 + c4 + c7 + c8 + c9)) };
                auto g{ hn::Max(hn::Max(hn::Max(hn::Abs(g1), hn::Abs(g2)), hn::Max(hn::Abs(g3), hn::Abs(g4))),
                                hn::Max(hn::Max(hn::Abs(g5), hn::Abs(g6)), hn::Max(hn::Abs(g7), hn::Abs(g8)))) };
                storeRow<kRing>(g * k(scale), df, gradient + x);
                break;
            }
        } else {
            auto c1{ hn::LoadU(df, prev2 + x - 2) };
            auto c2{ hn::LoadU(df, prev2 + x - 1) };
            auto c3{ hn::LoadU(df, prev2 + x) };
            auto c4{ hn::LoadU(df, prev2 + x + 1) };
            auto c5{ hn::LoadU(df, prev2 + x + 2) };
            auto c6{ hn::LoadU(df, prev + x - 2) };
            auto c7{ hn::LoadU(df, prev + x - 1) };
            auto c8{ hn::LoadU(df, prev + x) };
            auto c9{ hn::LoadU(df, prev + x + 1) };
            auto c10{ hn::LoadU(df, prev + x + 2) };
            auto c11{ hn::LoadU(df, cur + x - 2) };
            auto c12{ hn::LoadU(df, cur + x - 1) };
            auto c14{ hn::LoadU(df, cur + x + 1) };
            auto c15{ hn::LoadU(df, cur + x + 2) };
            auto c16{ hn::LoadU(df, next + x - 2) };
            auto c17{ hn::LoadU(df, next + x - 1) };
            auto c18{ hn::LoadU(df, next + x) };
            auto c19{ hn::LoadU(df, next + x + 1) };
            auto c20{ hn::LoadU(df, next + x + 2) };
            auto c21{ hn::LoadU(df, next2 + x - 2) };
            auto c22{ hn::LoadU(df, next2 + x - 1) };
            auto c23{ hn::LoadU(df, next2 + x) };
            auto c24{ hn::LoadU(df, next2 + x + 1) };
            auto c25{ hn::LoadU(df, next2 + x + 2) };

            gx = c5 + c25 + c4 + c24 + hn::MulAdd(k(2.0f), c10 + c20 + c9 + c19, k(3.0f) * (c15 + c14))
                - c2 - c22 - c1 - c21 - hn::MulAdd(k(2.0f), c7 + c17 + c6 + c16, k(3.0f) * (c12 + c11));
            gy = c1 + c5 + c6 + c10 + hn::MulAdd(k(2.0f), c2 + c4 + c7 + c9, k(3.0f) * (c3 + c8))
                - c16 - c20 - c21 - c25 - hn::MulAdd(k(2.0f), c17 + c19 + c22 + c24, k(3.0f) * (c18 + c23));
        }

        if (op != KIRSCH) {
            gx = gx * k(scale);
            gy = gy * k(scale);
            storeRow<kRing>(hn::Sqrt(hn::MulAdd(gx, gx, gy * gy)), df, gradient + x);
        }

        if (direction)
            storeRow<kRing>(directionBin(gy, gx), di, direction + x);
    }
}

// Upstream's row walk over the blur plane: mirrors the halo of the rows it reads and reflects at the bottom.
struct BlurRows {
    float *prev2, *prev, *cur, *next, *next2;
    int width, height, op;
    ptrdiff_t bgStride;

    BlurRows(float* blur, int w, int h, ptrdiff_t pitch, int o) noexcept
        : prev2{ blur + pitch * 2 }, prev{ blur + pitch }, cur{ blur }, next{ blur + pitch }, next2{ blur + pitch * 2 },
          width{ w }, height{ h }, op{ o }, bgStride{ pitch } {
        cur[-1] = cur[1];
        cur[width] = cur[width - 2];
        if (op == FDOG) {
            cur[-2] = cur[2];
            cur[width + 1] = cur[width - 3];
        }
    }

    void mirror() noexcept {  // before computing a row
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
    }

    void advance(int y) noexcept {  // after computing row y
        prev2 = prev;
        prev = cur;
        cur = next;
        if (op != FDOG) {
            next += (y < height - 2) ? bgStride : -bgStride;
        } else {
            next = next2;
            next2 += (y < height - 3) ? bgStride : -bgStride;
        }
    }
};

// Mode 1: the gradient plane.
static void detectEdge(float* blur, float* gradient, const int width, const int height, const ptrdiff_t bgStride, const int op,
                       const float scale) noexcept {
    BlurRows rows{ blur, width, height, bgStride, op };
    for (auto y{ 0 }; y < height; y++) {
        rows.mirror();
        edgeRow<false>(rows.prev2, rows.prev, rows.cur, rows.next, rows.next2, gradient, nullptr, width, op, scale);
        rows.advance(y);
        gradient += bgStride;
    }
}

// Mode 0 after the gradient. Upstream suppresses non-maxima into the blur plane, then runs hysteresis over it with a
// found array and binarizes the pixels it set to FLT_MAX. The result is the set of pixels >= t_l that are 8-connected
// to a pixel >= t_h, whatever the visiting order, so this path computes the same set with less memory traffic: NMS
// classifies each pixel straight into a byte map (0 none, 1 weak, 2 strong; zero border, so no bounds checks), the
// trace marks reached pixels 3, and the destination is written from the map.
static constexpr uint8_t kStrong{ 2 }, kEdge{ 3 };  // 0 = none, 1 = weak

// One row of upstream's NMS, classified into the map. up/down are the gradient rows above and below (mirrored at the
// plane's top and bottom, as upstream's halo rows), each with its columns -1 and width mirrored.
HWY_INLINE void nmsRow(const int* _direction, const float* up, const float* row, const float* down, uint8_t* mapRow,
                       const int width, const size_t mapStride, const float t_h, const float t_l) {
    const DF df;
    const DI di;
    const hn::Rebind<uint8_t, DF> du8;
    const int step{ static_cast<int>(hn::Lanes(df)) };
    const auto one{ hn::Set(di, 1) };
    mapRow[0] = 0;

    for (auto x{ 0 }; x < width; x += step) {
        auto direction{ hn::Load(di, _direction + x) };

        auto mask{ hn::RebindMask(df, hn::Eq(direction, hn::Zero(di))) };
        auto gradient{ hn::Max(hn::LoadU(df, row + x + 1), hn::LoadU(df, row + x - 1)) };
        auto result{ hn::IfThenElseZero(mask, gradient) };

        mask = hn::RebindMask(df, hn::Eq(direction, hn::Set(di, 1)));
        gradient = hn::Max(hn::LoadU(df, up + x + 1), hn::LoadU(df, down + x - 1));
        result = hn::Or(result, hn::IfThenElseZero(mask, gradient));

        mask = hn::RebindMask(df, hn::Eq(direction, hn::Set(di, 2)));
        gradient = hn::Max(hn::LoadU(df, up + x), hn::LoadU(df, down + x));
        result = hn::Or(result, hn::IfThenElseZero(mask, gradient));

        mask = hn::RebindMask(df, hn::Eq(direction, hn::Set(di, 3)));
        gradient = hn::Max(hn::LoadU(df, up + x - 1), hn::LoadU(df, down + x + 1));
        result = hn::Or(result, hn::IfThenElseZero(mask, gradient));

        gradient = hn::LoadU(df, row + x);
        const auto value{ hn::IfThenElse(hn::Ge(gradient, result), gradient, k(fltLowest)) };  // upstream's NMS output
        const auto weak{ hn::IfThenElseZero(hn::RebindMask(di, hn::Ge(value, k(t_l))), one) };
        const auto strong{ hn::IfThenElseZero(hn::RebindMask(di, hn::Ge(value, k(t_h))), one) };
        hn::StoreU(hn::DemoteTo(du8, hn::Add(weak, strong)), du8, mapRow + 1 + x);
    }
    std::fill(mapRow + 1 + width, mapRow + mapStride, uint8_t{ 0 });  // also clears what the last vector wrote past the row
}

// Mode 0: gradient, direction and NMS fused per row. NMS of row y-1 needs the gradient of rows y-2..y, so the gradient
// and the direction live in rings of 3 rows that stay in cache instead of full planes (gradientRing/directionRing:
// 3 rows of bgStride/stride, with the gradient rows' halo). Same arithmetic and mirroring as upstream's two passes.
static void detectEdgeNms(float* blur, float* gradientRing, int* directionRing, uint8_t* map, const int width, const int height,
                          const ptrdiff_t stride, const ptrdiff_t bgStride, const int op, const float scale, const size_t mapStride,
                          const float t_h, const float t_l) noexcept {
    const auto g = [&](int y) { return gradientRing + bgStride * (y % 3); };
    const auto dir = [&](int y) { return directionRing + stride * (y % 3); };
    std::fill_n(map, mapStride, uint8_t{ 0 });
    std::fill_n(map + mapStride * (height + 1), mapStride, uint8_t{ 0 });

    BlurRows rows{ blur, width, height, bgStride, op };
    for (auto y{ 0 }; y < height; y++) {  // height >= 3 in mode 0 (checked at creation)
        rows.mirror();
        float* gy{ g(y) };
        edgeRow<true>(rows.prev2, rows.prev, rows.cur, rows.next, rows.next2, gy, dir(y), width, op, scale);
        gy[-1] = gy[1];
        gy[width] = gy[width - 2];
        rows.advance(y);
        if (y >= 1)  // row y-1; above row 0 is row 1 (upstream's halo row -1)
            nmsRow(dir(y - 1), g(y == 1 ? 1 : y - 2), g(y - 1), gy, map + mapStride * y, width, mapStride, t_h, t_l);
    }
    // Last row; below it is row height-2 (upstream's halo row height).
    nmsRow(dir(height - 1), g(height - 2), g(height - 1), g(height - 2), map + mapStride * height, width, mapStride, t_h, t_l);
}

// Depth-first from every strong pixel through weak and strong ones, as upstream's hysteresis but on the byte map.
// (Classifying the 8 neighbours with three 4-byte loads instead of 8 branches was tried: 3-5% slower on Zen 3.)
static void traceEdges(uint8_t* map, std::vector<int>& stack, const int width, const int height, const size_t mapStride) noexcept {
    const hn::CappedTag<uint8_t, 64> du;
    const int step{ static_cast<int>(hn::Lanes(du)) };
    const int s{ static_cast<int>(mapStride) };
    const int offsets[]{ -s - 1, -s, -s + 1, -1, 1, s - 1, s, s + 1 };
    stack.clear();

    for (auto y{ 1 }; y <= height; y++) {
        uint8_t* row{ map + mapStride * y };
        for (auto x{ 1 }; x <= width; x += step) {
            // Reads up to 64 bytes past x: inside the map, because the pitch is at least 64 and the bottom border row exists.
            if (hn::AllFalse(du, hn::Eq(hn::LoadU(du, row + x), hn::Set(du, kStrong))))
                continue;
            const auto stop{ std::min(x + step, width + 1) };
            for (auto i{ x }; i < stop; i++) {
                if (row[i] != kStrong)
                    continue;
                row[i] = kEdge;
                stack.push_back(s * y + i);
                while (!stack.empty()) {
                    const auto p{ stack.back() };
                    stack.pop_back();
                    for (const auto o : offsets) {
                        auto& m{ map[p + o] };
                        if (static_cast<uint8_t>(m - 1) < 2) {  // weak or strong, not yet reached
                            m = kEdge;
                            stack.push_back(p + o);
                        }
                    }
                }
            }
        }
    }
}

template<typename pixel_t>
static void binarizeEdgeMap(const uint8_t* map, pixel_t* dstp, const int width, const int height, const size_t mapStride,
                            const ptrdiff_t dstStride, const int peak) noexcept {
    const DF df;
    const DI di;
    const hn::Rebind<uint8_t, DF> du8;
    const int step{ static_cast<int>(hn::Lanes(df)) };

    for (auto y{ 0 }; y < height; y++) {
        const uint8_t* row{ map + mapStride * (y + 1) + 1 };
        for (auto x{ 0 }; x < width; x += step) {
            const auto edge{ hn::Eq(hn::PromoteTo(di, hn::LoadU(du8, row + x)), hn::Set(di, kEdge)) };

            if constexpr (std::is_same_v<pixel_t, float>) {
                hn::Stream(hn::IfThenElse(hn::RebindMask(df, edge), hn::Set(df, 1.0f), hn::Zero(df)), df, dstp + x);
            } else {
                // 8-bit always writes 255, like upstream.
                const auto value{ hn::Set(di, std::is_same_v<pixel_t, uint8_t> ? 255 : peak) };
                const hn::Rebind<pixel_t, DF> dp;
                hn::StoreU(hn::DemoteTo(dp, hn::IfThenElseZero(edge, value)), dp, dstp + x);
            }
        }
        dstp += dstStride;
    }
}

template<typename pixel_t, bool clampFP = true>
static void discretizeGM(const float* _srcp, pixel_t* dstp, const int width, const int height, const ptrdiff_t srcStride, const ptrdiff_t dstStride,
                         const int peak) noexcept {
    const DF df;
    const DI di;
    const int step{ static_cast<int>(hn::Lanes(df)) };

    for (auto y{ 0 }; y < height; y++) {
        for (auto x{ 0 }; x < width; x += step) {
            auto srcp{ hn::LoadU(df, _srcp + x) };

            if constexpr (std::is_same_v<pixel_t, float>) {
                if constexpr (clampFP)
                    hn::Stream(hn::Min(hn::Max(srcp, hn::Zero(df)), hn::Set(df, 1.0f)), df, dstp + x);
                else
                    hn::Stream(srcp, df, dstp + x);
            } else {
                // Saturating demotion, then (16-bit) the peak; truncation as VCL's truncatei.
                const hn::Rebind<pixel_t, DF> dp;
                auto result{ hn::DemoteTo(dp, hn::ConvertInRangeTo(di, srcp + hn::Set(df, 0.5f))) };
                if constexpr (std::is_same_v<pixel_t, uint16_t>)
                    result = hn::Min(result, hn::Set(dp, static_cast<uint16_t>(peak)));
                hn::StoreU(result, dp, dstp + x);
            }
        }

        _srcp += srcStride;
        dstp += dstStride;
    }
}

template<typename pixel_t>
void filter_simd(const VSFrame* src, VSFrame* dst, const TCannyData* const VS_RESTRICT d, TCannyScratch* scratch, const VSAPI* vsapi) noexcept {
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

            if (d->radiusH[plane] && d->radiusV[plane])
                if constexpr (kBlurRing<pixel_t>)
                    gaussianBlurRing(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->radiusV[plane],
                                     d->weightsH[plane].data(), d->weightsV[plane].data(), scratch->blurRows.get(), tcannyBlurRowsPitch(width));
                else
                    gaussianBlur(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->radiusV[plane],
                                 d->weightsH[plane].data(), d->weightsV[plane].data());
            else if (d->radiusH[plane])
                gaussianBlurH(srcp, gradient, blur, width, height, stride, bgStride, d->radiusH[plane], d->weightsH[plane].data());
            else if (d->radiusV[plane])
                if constexpr (kBlurRing<pixel_t>)
                    gaussianBlurVRing(srcp, blur, width, height, stride, bgStride, d->radiusV[plane], d->weightsV[plane].data(),
                                      scratch->blurRows.get(), tcannyBlurRowsPitch(width));
                else
                    gaussianBlurV(srcp, blur, width, height, stride, bgStride, d->radiusV[plane], d->weightsV[plane].data());
            else
                copyPlane(srcp, blur, width, height, stride, bgStride);

            if (d->mode == 0) {
                const auto mapStride{ tcannyEdgeMapStride(width) };
                detectEdgeNms(blur, gradient, direction, scratch->edgeMap.get(), width, height, stride, bgStride, d->op, d->scale,
                              mapStride, d->t_h, d->t_l);
                traceEdges(scratch->edgeMap.get(), scratch->edgeStack, width, height, mapStride);
                binarizeEdgeMap(scratch->edgeMap.get(), dstp, width, height, mapStride, stride, d->peak);
            } else if (d->mode == 1) {
                detectEdge(blur, gradient, width, height, bgStride, d->op, d->scale);
            }

            if (d->mode == 1)
                discretizeGM(gradient, dstp, width, height, bgStride, stride, d->peak);
            else if (d->mode == -1)
                discretizeGM<pixel_t, false>(blur, dstp, width, height, bgStride, stride, d->peak);
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace tcanny_simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace tcanny_simd {
namespace {
struct Entry {
    int opt;  // the `opt` value that names this target
    int64_t target;
    TCannyFilter filter[3];  // uint8_t, uint16_t, float
};

#define TCANNY_ENTRY(OPT, TARGET, NS) { OPT, TARGET, { NS::filter_simd<uint8_t>, NS::filter_simd<uint16_t>, NS::filter_simd<float> } }
// Best first: opt=0 takes the first one the CPU supports.
constexpr Entry kEntries[]{
#if HWY_TARGETS & HWY_AVX3
    TCANNY_ENTRY(4, HWY_AVX3, N_AVX3),
#endif
#if HWY_TARGETS & HWY_AVX2
    TCANNY_ENTRY(3, HWY_AVX2, N_AVX2),
#endif
#if HWY_TARGETS & HWY_SSE2
    TCANNY_ENTRY(2, HWY_SSE2, N_SSE2),
#endif
// The two NEON targets differ only in AES, which these kernels do not use; a build whose baseline has AES (Apple)
// compiles only HWY_NEON.
#if HWY_TARGETS & HWY_NEON
    TCANNY_ENTRY(0, HWY_NEON, N_NEON),
#endif
#if HWY_TARGETS & HWY_NEON_WITHOUT_AES
    TCANNY_ENTRY(0, HWY_NEON_WITHOUT_AES, N_NEON_WITHOUT_AES),
#endif
    { -1, 0, { nullptr, nullptr, nullptr } },  // never supported; keeps the array non-empty
};
#undef TCANNY_ENTRY
}  // namespace
}  // namespace tcanny_simd

TCannyFilter tcannySelectSimd(int bytesPerSample, int opt) {
    using tcanny_simd::kEntries;
    const int index{ bytesPerSample == 1 ? 0 : bytesPerSample == 2 ? 1 : 2 };
    const int64_t supported{ hwy::SupportedTargets() };
#if HWY_ARCH_X86
    for (const auto& e : kEntries) {
        if (opt != 0 && e.opt != opt) continue;
        if (supported & e.target) return e.filter[index];
        if (opt != 0) break;
    }
    if (opt == 0) return nullptr;  // no SIMD target: scalar
    throw std::runtime_error("requested opt is not supported by this CPU");
#else
    if (opt > 1) throw std::runtime_error("SIMD opt requires an x86 CPU");
    for (const auto& e : kEntries)
        if (supported & e.target) return e.filter[index];
    return nullptr;
#endif
}
#endif  // HWY_ONCE
