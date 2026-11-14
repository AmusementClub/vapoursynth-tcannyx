// Profiling counters, compiled in only with -Dprofiling=true (defines TCANNY_PROF) and printed to stderr when the
// environment variable TCANNY_PROF is set. Without the define every type below is an empty no-op.
#pragma once
#include <cstdint>

#ifdef TCANNY_PROF
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace tcprof {
inline bool enabled() { static const bool e = std::getenv("TCANNY_PROF") != nullptr; return e; }
inline uint64_t now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Acc {
    std::atomic<uint64_t> ns{0}, cnt{0};
    void add(uint64_t d) { ns.fetch_add(d, std::memory_order_relaxed); cnt.fetch_add(1, std::memory_order_relaxed); }
};
struct Timer {
    Acc *a; uint64_t t0;
    explicit Timer(Acc &acc) : a(enabled() ? &acc : nullptr), t0(a ? now() : 0) {}
    void stop() { if (a) { a->add(now() - t0); a = nullptr; } }
    ~Timer() { stop(); }
};
inline void line(const char *name, const Acc &a, uint64_t frames) {
    const double total_ms = a.ns.load() / 1e6;
    std::fprintf(stderr, "  %-34s calls=%-8llu total=%10.2f ms  per-frame=%8.4f ms  per-call=%9.4f ms\n", name,
                 (unsigned long long)a.cnt.load(), total_ms, frames ? total_ms / frames : 0.0, a.cnt.load() ? total_ms / a.cnt.load() : 0.0);
}
// Per-frame hysteresis statistics (batches, passes executed) of one filter instance.
struct HystStats {
    std::mutex m; std::vector<std::pair<uint32_t, uint32_t>> perFrame;
    void add(uint32_t batches, uint32_t passes) { if (!enabled()) return; std::lock_guard<std::mutex> g(m); perFrame.emplace_back(batches, passes); }
    void print(const char *tag) {
        std::lock_guard<std::mutex> g(m);
        if (perFrame.empty()) return;
        uint64_t sb = 0, sp = 0; uint32_t minb = ~0u, maxb = 0, minp = ~0u, maxp = 0;
        for (auto &p : perFrame) { sb += p.first; sp += p.second; minb = std::min(minb, p.first); maxb = std::max(maxb, p.first); minp = std::min(minp, p.second); maxp = std::max(maxp, p.second); }
        std::fprintf(stderr, "  %s hysteresis frames=%zu batches/frame mean=%.2f min=%u max=%u | passes-executed/frame mean=%.1f min=%u max=%u | first frames (batches,passes):",
                     tag, perFrame.size(), double(sb) / perFrame.size(), minb, maxb, double(sp) / perFrame.size(), minp, maxp);
        for (size_t i = 0; i < perFrame.size() && i < 8; ++i) std::fprintf(stderr, " (%u,%u)", perFrame[i].first, perFrame[i].second);
        std::fprintf(stderr, "\n");
        // Submissions per frame = batches + 1 (the finalize submission). The first kWarmup frames still have an empty depth
        // hint, so the percentiles are given for all frames and for the steady state after them.
        percentiles(tag, "all", 0);
        percentiles(tag, "steady", kWarmup);
    }
    static constexpr size_t kWarmup = 16;
    void percentiles(const char *tag, const char *label, size_t skip) {
        if (perFrame.size() <= skip) return;
        std::vector<uint32_t> b;
        for (size_t i = skip; i < perFrame.size(); ++i) b.push_back(perFrame[i].first + 1);
        std::sort(b.begin(), b.end());
        const auto at = [&](double q) { return b[std::min(b.size() - 1, size_t(q * b.size()))]; };
        std::fprintf(stderr, "  %s submissions/frame (%s, n=%zu): p50=%u p90=%u p99=%u max=%u\n", tag, label, b.size(), at(0.50), at(0.90), at(0.99), b.back());
    }
};
} // namespace tcprof
#else
namespace tcprof {
constexpr bool enabled() { return false; }
struct Acc {};
struct Timer {
    explicit Timer(Acc &) {}
    void stop() {}
};
inline void line(const char *, const Acc &, uint64_t) {}
struct HystStats {
    void add(uint32_t, uint32_t) {}
    void print(const char *) {}
};
} // namespace tcprof
#endif
