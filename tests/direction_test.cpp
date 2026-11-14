// Mode-0 direction bins: the fast path (directionBin) must equal upstream's atan2 path (directionBinExact) lane for lane
// on every compiled Highway target this CPU supports. Inputs: random magnitudes over 1e-35..1e35, dense sweeps across
// the four bin boundaries, every integer pair in [-1024, 1024]^2 (and halves), and special values.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "direction_test.cpp"
#include "cpu_simd_targets.h"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "cpu_direction-inl.h"

HWY_BEFORE_NAMESPACE();
namespace tcanny_simd {
namespace HWY_NAMESPACE {

struct Counts {
    uint64_t lanes, mismatches;
};

// Compares both paths on gy[i], gx[i]; prints the first few mismatches.
static void check(const std::vector<float>& gy, const std::vector<float>& gx, Counts& c) {
    const DF df;
    const DI di;
    const size_t n{ hn::Lanes(df) };
    std::vector<int32_t> fast(n), exact(n);
    for (size_t i{ 0 }; i + n <= gy.size(); i += n) {
        const VF y{ hn::LoadU(df, gy.data() + i) };
        const VF x{ hn::LoadU(df, gx.data() + i) };
        hn::StoreU(directionBin(y, x), di, fast.data());
        hn::StoreU(directionBinExact(y, x), di, exact.data());
        for (size_t j{ 0 }; j < n; j++) {
            if (fast[j] != exact[j] && c.mismatches++ < 10)
                std::printf("  %s: gy=%a gx=%a fast=%d exact=%d\n", hwy::TargetName(HWY_TARGET), gy[i + j], gx[i + j], fast[j], exact[j]);
        }
        c.lanes += n;
    }
}

// Returns the number of mismatching lanes.
uint64_t RunDirectionTest() {
    Counts c{};
    std::vector<float> gy, gx;
    const auto flush = [&] {
        while (gy.size() % 16) { gy.push_back(0.0f); gx.push_back(0.0f); }
        check(gy, gx, c);
        gy.clear();
        gx.clear();
    };
    const auto add = [&](double y, double x) {
        gy.push_back(static_cast<float>(y));
        gx.push_back(static_cast<float>(x));
        if (gy.size() == 1 << 16) flush();
    };

    // Random angles and magnitudes, uniform lanes and whole vectors of one magnitude.
    std::mt19937_64 rng{ 20261115 };
    std::uniform_real_distribution<double> angle(-M_PI, M_PI), logMag(-35.0, 35.0), unit(0.0, 1.0);
    for (int i{ 0 }; i < 4000000; i++) {
        const double r{ std::pow(10.0, logMag(rng)) }, a{ angle(rng) };
        add(r * std::sin(a), r * std::cos(a));
    }
    // Right at the boundaries pi/8 + k pi/4, both quadrant halves, offsets from 1e-9 to 1e-2 rad and a few float ulps.
    for (int b{ 0 }; b < 8; b++) {
        const double phi{ M_PI / 8 + b * M_PI / 4 - M_PI };
        for (int e{ -9 }; e <= -2; e++) {
            for (int i{ 0 }; i < 20000; i++) {
                const double off{ std::pow(10.0, e + unit(rng)) * (i % 2 ? 1 : -1) };
                const double r{ std::pow(10.0, logMag(rng) * 0.3) };
                add(r * std::sin(phi + off), r * std::cos(phi + off));
            }
        }
        for (int i{ 0 }; i < 20000; i++) {
            const double r{ 1 + 999 * unit(rng) };
            float y{ static_cast<float>(r * std::sin(phi)) }, x{ static_cast<float>(r * std::cos(phi)) };
            for (int s{ 0 }; s < (i % 7); s++) y = std::nextafter(y, i % 2 ? 1e30f : -1e30f);
            add(y, x);
        }
    }
    // Integer gradients (8-bit content with Tritical/Sobel-like operators) and halves (Prewitt).
    for (int y{ -1024 }; y <= 1024; y++)
        for (int x{ -1024 }; x <= 1024; x++)
            add(y, x);
    for (int y{ -512 }; y <= 512; y++)
        for (int x{ -512 }; x <= 512; x++)
            add(y * 0.5, x * 0.5);
    flush();
    // Special values, in every pairing.
    const float inf{ std::numeric_limits<float>::infinity() }, nan{ std::numeric_limits<float>::quiet_NaN() };
    const float specials[]{ 0.0f, -0.0f, 1.0f, -1.0f, 1e-30f, 9e-31f, 1e-38f, 1e-45f, -1e-45f, 1e30f, 1.1e30f, 3.4e38f, -3.4e38f,
                            inf, -inf, nan, 0.41421356f, 2.4142135f, 1e-20f, 1e20f };
    for (float y : specials)
        for (float x : specials)
            for (int lane{ 0 }; lane < 16; lane++) {  // alone in a vector and mixed with ordinary lanes
                gy.push_back(lane == 0 ? y : 1.0f);
                gx.push_back(lane == 0 ? x : 3.0f);
            }
    for (float y : specials)
        for (float x : specials) {
            gy.push_back(y);
            gx.push_back(x);
        }
    flush();

    std::printf("%s: %llu lanes, %llu mismatches\n", hwy::TargetName(HWY_TARGET), static_cast<unsigned long long>(c.lanes),
                static_cast<unsigned long long>(c.mismatches));
    return c.mismatches;
}

}  // namespace HWY_NAMESPACE
}  // namespace tcanny_simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace tcanny_simd {
HWY_EXPORT(RunDirectionTest);
static uint64_t runDirectionTest() { return HWY_DYNAMIC_DISPATCH(RunDirectionTest)(); }
}  // namespace tcanny_simd

int main() {
    uint64_t mismatches{ 0 };
    int targets{ 0 };
    for (const int64_t target : hwy::SupportedAndGeneratedTargets()) {
        hwy::SetSupportedTargetsForTest(target);
        mismatches += tcanny_simd::runDirectionTest();
        targets++;
    }
    hwy::SetSupportedTargetsForTest(0);
    std::printf("%d targets, %llu mismatches\n", targets, static_cast<unsigned long long>(mismatches));
    return mismatches == 0 && targets > 0 ? 0 : 1;
}
#endif
