#pragma once

#define VS_USE_API_43
#include <VapourSynth4.h>
#include <VSHelper4.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class TCannyBackend {
    Cpu,
    Vulkan,
    Cuda,
};

enum TCannyOperator {
    TRITICAL = 0,
    PREWITT = 1,
    SOBEL = 2,
    SCHARR = 3,
    KROON = 4,
    KIRSCH = 5,
    FDOG = 6,
};

struct TCannyScratch;

struct TCannyData {
    VSNode *node{};
    const VSVideoInfo *vi{};
    TCannyBackend backend{TCannyBackend::Cpu};
    bool gpuOutput{};
    int mode{};
    int op{PREWITT};
    float t_h{8.0f};
    float t_l{1.0f};
    float scale{1.0f};
    int opt{};
    int radiusAlign{};
    size_t alignment{64};
    void (*filter)(const VSFrame *, VSFrame *, const TCannyData *, TCannyScratch *, const VSAPI *) noexcept{};
    int peak{255};
    std::array<bool, 3> process{true, true, true};
    std::array<float, 3> sigmaH{};
    std::array<float, 3> sigmaV{};
    std::array<int, 3> radiusH{};
    std::array<int, 3> radiusV{};
    std::array<std::vector<float>, 3> weightsH;
    std::array<std::vector<float>, 3> weightsV;
    void *vulkanState{};
    void *cpuPool{};
    void *cudaState{};
};

void tcannySetupCpu(TCannyData *data);
void tcannyDestroyCpuPool(void *pool) noexcept;
void tcannyProcessCpu(const VSFrame *src, VSFrame *dst, const TCannyData *data, const VSAPI *vsapi);

bool tcannyCudaCompiled() noexcept;
void *tcannyCreateCudaState(std::string &error) noexcept;
void tcannyDestroyCudaState(void *state) noexcept;
void tcannyCudaProfDump() noexcept; // prints the profiling counters when built with -Dprofiling (no-op otherwise)
VSFrame *tcannyProcessCuda(const VSFrame *src, const TCannyData *data,
                           VSCore *core, const VSAPI *vsapi, std::string &error) noexcept;
void *tcannyCreateVulkanState(VSCore *core, const VSAPI *vsapi, const TCannyData *data, std::string &error) noexcept;
void tcannyDestroyVulkanState(void *state) noexcept;
VSFrame *tcannyProcessVulkan(const VSFrame *src, const TCannyData *data,
                             VSCore *core, const VSAPI *vsapi, std::string &error) noexcept;

