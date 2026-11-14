// GPL-3.0-or-later; adapted from HomeOfVapourSynthEvolution/VapourSynth-TCanny.
//
// Mode-0 gradient direction bins for cpu_simd.cpp, per Highway target (include after <hwy/highway.h>, once per target).
// directionBinExact is upstream's computation: VCL's atan2, folded to [0, pi], rounded to a multiple of pi/4.
// directionBin gives the same bins without the atan2 when every lane is clearly inside one; tests/direction_test.cpp
// checks the two against each other on every compiled target.

// Per-target include guard (Highway's convention for -inl headers).
#if defined(TCANNY_CPU_DIRECTION_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef TCANNY_CPU_DIRECTION_INL_H_
#undef TCANNY_CPU_DIRECTION_INL_H_
#else
#define TCANNY_CPU_DIRECTION_INL_H_
#endif

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace tcanny_simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// At most 16 floats: the x loops run in whole vectors up to the stride and rely on the 64-byte frame alignment and the
// radiusAlign (16-float) halo of the scratch rows.
using DF = hn::CappedTag<float, 16>;
using DI = hn::RebindToSigned<DF>;
using VF = hn::Vec<DF>;
using VI = hn::Vec<DI>;
using MF = hn::Mask<DF>;

// Same values as M_PIF and M_1_PIF (cpu_reference.h).
constexpr float kPi{ 3.14159265358979323846f };
constexpr float kInvPi{ 0.318309886183790671538f };

HWY_INLINE VF k(float v) { return hn::Set(DF(), v); }

// VCL's if_add: a masked add on AVX-512, a + (mask & b) elsewhere.
HWY_INLINE VF ifAdd(MF m, VF a, VF b) {
#if HWY_TARGET == HWY_AVX3
    return hn::IfThenElse(m, hn::Add(a, b), a);
#else
    return hn::Add(a, hn::IfThenElseZero(m, b));
#endif
}

// VCL's atan_f<VTYPE, 1> (vectormath_trig.h): same constants and operation order, so the mode-0 direction bins match.
// Ported from the Vector Class Library, (c) 2014-2020 Agner Fog, Apache License 2.0.
HWY_INLINE VF atan2Vcl(VF y, VF x) {
    const DF df;
    const DI di;
    const VF x1{ hn::Abs(x) };
    const VF y1{ hn::Abs(y) };
    const MF swapxy{ hn::Gt(y1, x1) };
    VF x2{ hn::IfThenElse(swapxy, y1, x1) };
    VF y2{ hn::IfThenElse(swapxy, x1, y1) };

    const MF bothInfinite{ hn::And(hn::IsInf(x), hn::IsInf(y)) };
    if (!hn::AllFalse(df, bothInfinite)) {
        const VF mone{ hn::Set(df, -1.0f) };
        x2 = hn::IfThenElse(bothInfinite, hn::And(x2, mone), x2);
        y2 = hn::IfThenElse(bothInfinite, hn::And(y2, mone), y2);
    }

    const VF t{ hn::Div(y2, x2) };  // 0/0 gives NaN, replaced below
    const MF notsmal{ hn::Ge(t, hn::Set(df, static_cast<float>(1.41421356237309504880 - 1.))) };
    const VF a{ ifAdd(notsmal, t, hn::Set(df, -1.0f)) };
    const VF b{ ifAdd(notsmal, hn::Set(df, 1.0f), t) };
    const VF s{ hn::IfThenElseZero(notsmal, hn::Set(df, static_cast<float>(0.785398163397448309616))) };
    const VF z{ hn::Div(a, b) };
    const VF zz{ hn::Mul(z, z) };

    // polynomial_3(zz, P0, P1, P2, P3)
    const VF zz2{ hn::Mul(zz, zz) };
    VF re{ hn::MulAdd(hn::MulAdd(hn::Set(df, 8.05374449538E-2f), zz, hn::Set(df, -1.38776856032E-1f)), zz2,
                      hn::MulAdd(hn::Set(df, 1.99777106478E-1f), zz, hn::Set(df, -3.33329491539E-1f))) };
    re = hn::Add(hn::MulAdd(re, hn::Mul(zz, z), z), s);

    re = hn::IfThenElse(swapxy, hn::Sub(hn::Set(df, static_cast<float>(1.57079632679489661923)), re), re);
    re = hn::IfThenElse(hn::Eq(hn::Or(x, y), hn::Zero(df)), hn::Zero(df), re);  // atan2(0, +0) = 0
    const MF signX{ hn::RebindMask(df, hn::Lt(hn::BitCast(di, x), hn::Zero(di))) };  // also for x = -0
    re = hn::IfThenElse(signX, hn::Sub(hn::Set(df, static_cast<float>(3.14159265358979323846)), re), re);
    return hn::Xor(re, hn::And(y, hn::Set(df, -0.0f)));  // sign_combine(re, y)
}

// Upstream: bin = trunc(atan2(gy, gx) folded to [0, pi] * 4 / pi + 0.5), 4 wraps to 0.
// 0 = horizontal gradient, 1 = gx and gy of the same sign, 2 = vertical, 3 = opposite signs.
HWY_INLINE VI directionBinExact(VF gy, VF gx) {
    const DF df;
    const DI di;
    auto dr{ atan2Vcl(gy, gx) };
    dr = ifAdd(hn::Lt(dr, hn::Zero(df)), dr, k(kPi));
    const auto bin{ hn::ConvertInRangeTo(di, hn::MulAdd(dr, k(4.0f * kInvPi), k(0.5f))) };
    return hn::IfThenZeroElse(hn::Ge(bin, hn::Set(di, 4)), bin);
}

// The bin from |gy| / |gx| against tan(pi/8) and tan(3pi/8) with a relative margin of 1e-3 (about 3.5e-4 rad; the
// float atan2 path is accurate to about 1e-6 in the scaled angle), so a lane decided here gets the bin upstream gets.
// Both gradients exactly 0 give bin 0 upstream for every sign combination (traced through atan_f), so flat regions stay
// here. Anything else near a boundary, non-finite, below 1e-30 (denormal products) or above 1e30 (overflowing products)
// sends the whole vector through directionBinExact; on natural content that is about 1% of vectors.
HWY_INLINE VI directionBin(VF gy, VF gx) {
    const DF df;
    const DI di;
    constexpr double kTan1{ 0.41421356237309504880 };  // tan(pi/8)
    constexpr double kTan3{ 2.41421356237309504880 };  // tan(3pi/8)
    constexpr double kMargin{ 1e-3 };
    const VF ax{ hn::Abs(gx) };
    const VF ay{ hn::Abs(gy) };
    const VF sum{ hn::Add(ax, ay) };  // NaN stays NaN, Inf stays Inf: both fail the range test
    const MF zero{ hn::Eq(sum, hn::Zero(df)) };
    const MF inRange{ hn::Or(zero, hn::And(hn::Ge(sum, k(1e-30f)), hn::Le(sum, k(1e30f)))) };
    const MF horizontal{ hn::Or(zero, hn::Lt(ay, hn::Mul(k(static_cast<float>(kTan1 * (1 - kMargin))), ax))) };
    const MF vertical{ hn::Gt(ay, hn::Mul(k(static_cast<float>(kTan3 * (1 + kMargin))), ax)) };
    const MF diagonal{ hn::And(hn::Gt(ay, hn::Mul(k(static_cast<float>(kTan1 * (1 + kMargin))), ax)),
                               hn::Lt(ay, hn::Mul(k(static_cast<float>(kTan3 * (1 - kMargin))), ax))) };
    if (!hn::AllTrue(df, hn::And(inRange, hn::Or(hn::Or(horizontal, vertical), diagonal))))
        return directionBinExact(gy, gx);
    // Diagonal lanes have gx, gy != 0: same sign is bin 1, opposite signs bin 3.
    const auto sameSign{ hn::Ge(hn::BitCast(di, hn::Xor(gx, gy)), hn::Zero(di)) };
    const VI diag{ hn::IfThenElse(sameSign, hn::Set(di, 1), hn::Set(di, 3)) };
    return hn::IfThenZeroElse(hn::RebindMask(di, horizontal), hn::IfThenElse(hn::RebindMask(di, vertical), hn::Set(di, 2), diag));
}

}  // namespace HWY_NAMESPACE
}  // namespace tcanny_simd
HWY_AFTER_NAMESPACE();

#endif  // per-target include guard
