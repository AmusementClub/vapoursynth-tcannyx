// Batch sizing for the GPU hysteresis loops (shared by the Vulkan and CUDA backends).
//
// A hysteresis pass advances a promotion by a bounded distance, so a frame needs about as many passes as the depth of its
// longest weak chain. Passes are cheap (about 2 us each on the validation GPU) but every blocking round trip to read the
// convergence flag costs about as much as 30 passes, so the first batch is sized from what recent frames needed.
//
// Convergence protocol: each promoting invocation does atomicMax(flag, passIndex + 1), with the flag zeroed at the start
// of the batch. After a batch of `batch` passes, value v means pass v - 1 was the last that promoted anything:
//   v < batch  -> pass v (at least) promoted nothing, so the state is a fixed point and the frame is done; its depth
//                 (passes needed before the idle one) is passesBefore + v
//   v == batch -> the last pass still promoted: run another batch
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

class HysteresisDepthHint {
public:
    static constexpr uint32_t kMinBatch = 2;
    static constexpr uint32_t kMaxBatch = 256;

    // Size of the first batch of the next frame: enough for the deepest of the last few frames plus the idle pass that
    // proves convergence, rounded up to a multiple of 2 (the tile pass needs only a handful of passes and an idle one costs about as much as a working one).
    uint32_t firstBatch() const noexcept {
        uint32_t deepest = 0;
        for (const auto &depth : recent_) deepest = std::max(deepest, depth.load(std::memory_order_relaxed));
        const uint32_t wanted = (deepest + 1 + 1) / 2 * 2;
        return std::clamp(wanted, kMinBatch, kMaxBatch);
    }

    void record(uint32_t depth) noexcept {
        recent_[next_.fetch_add(1, std::memory_order_relaxed) % recent_.size()].store(depth, std::memory_order_relaxed);
    }

    // Batch size after one that did not converge.
    static uint32_t grow(uint32_t batch) noexcept { return std::min(batch * 2, kMaxBatch); }

private:
    std::array<std::atomic<uint32_t>, 4> recent_{};
    std::atomic<uint32_t> next_{0};
};
