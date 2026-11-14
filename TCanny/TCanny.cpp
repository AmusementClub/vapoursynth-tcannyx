/*
 * VapourSynth TCanny, derived from the GPL-3.0-or-later upstream filter.
 * The CPU implementation is intentionally kept as the numerical reference
 * for the Vulkan and CUDA backends.
 */
#include "TCanny.h"
#ifdef TCANNY_WITH_CUDA
#include <VSVulkan4.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <cstring>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

std::vector<float> gaussianWeights(float sigma, int &radius) {
    if (sigma <= 0.0f) {
        radius = 0;
        return {};
    }
    const int diameter = std::max(static_cast<int>(sigma * 3.0f + 0.5f), 1) * 2 + 1;
    radius = diameter / 2;
    std::vector<float> weights(static_cast<size_t>(diameter));
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const float value = std::exp(-(k * k) / (2.0f * sigma * sigma));
        weights[static_cast<size_t>(k + radius)] = value;
        sum += value;
    }
    for (float &value : weights)
        value /= sum;
    return weights;
}

bool isConstantSupported(const VSVideoInfo *vi) {
    return vi && vsh::isConstantVideoFormat(vi) &&
           ((vi->format.sampleType == stInteger && vi->format.bitsPerSample >= 8 && vi->format.bitsPerSample <= 16) ||
            (vi->format.sampleType == stFloat && vi->format.bitsPerSample == 32));
}

const VSFrame *VS_CC tcannyGetFrame(int n, int activationReason, void *instanceData, void **, VSFrameContext *frameCtx,
                                     VSCore *core, const VSAPI *vsapi) {
    auto *d = static_cast<TCannyData *>(instanceData);
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    if (activationReason != arAllFramesReady)
        return nullptr;
    const VSFrame *src = nullptr;
#if VAPOURSYNTH_API_MINOR >= 3 && defined(TCANNY_WITH_CUDA)
    if (d->backend == TCannyBackend::Cuda) {
        char exportError[512]{};
        const VSVULKANAPI *vulkan = vsapi->getVulkanAPI();
        if (!vulkan) {
            vsapi->setFilterError("TCanny: CUDA input export failed: VapourSynth Vulkan API is unavailable", frameCtx);
            return nullptr;
        }
        src = vulkan->getExportableFrameFilter(n, d->node, frameCtx, exportError, sizeof(exportError));
        if (!src) {
            vsapi->setFilterError((std::string("TCanny: CUDA input export failed: ") +
                                   (exportError[0] ? exportError : "no exportable input frame")).c_str(), frameCtx);
            return nullptr;
        }
    } else
#endif
        src = vsapi->getFrameFilter(n, d->node, frameCtx);
    if (!src) {
        vsapi->setFilterError("TCanny: failed to get input frame", frameCtx);
        return nullptr;
    }
    if (d->gpuOutput) {
        VSFrame *dst = nullptr;
        std::string error;
        const bool cuda = d->backend == TCannyBackend::Cuda;
        if (cuda) {
            if (!tcannyCudaCompiled()) error = "CUDA backend was not compiled in this build";
            else dst = tcannyProcessCuda(src, d, core, vsapi, error);
        } else {
            dst = tcannyProcessVulkan(src, d, core, vsapi, error);
        }
        if (dst && !error.empty()) {
            // A frame together with an error is a failure; never return both.
            vsapi->freeFrame(dst);
            dst = nullptr;
        }
        if (!dst && error.empty())
            error = cuda ? "CUDA backend failed to produce an output frame" : "Vulkan backend is unavailable for this frame";
        if (!error.empty())
            vsapi->setFilterError(("TCanny: " + error).c_str(), frameCtx);
        vsapi->freeFrame(src);
        return dst;
    }
    const VSFrame *planes[]{d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src};
    const int planeMap[]{0, 1, 2};
    VSFrame *dst = vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, planes, planeMap, src, core);
    if (!dst) {
        vsapi->setFilterError("TCanny: failed to allocate output frame", frameCtx);
        vsapi->freeFrame(src);
        return nullptr;
    }
    try {
        tcannyProcessCpu(src, dst, d, vsapi);
    } catch (const std::exception &e) {
        vsapi->setFilterError(("TCanny: " + std::string(e.what())).c_str(), frameCtx);
        vsapi->freeFrame(src);
        vsapi->freeFrame(dst);
        return nullptr;
    }
    vsapi->freeFrame(src);
    return dst;
}

void VS_CC tcannyFree(void *instanceData, VSCore *, const VSAPI *vsapi) {
    auto *d = static_cast<TCannyData *>(instanceData);
    if (d->backend == TCannyBackend::Cuda) tcannyCudaProfDump();
    tcannyDestroyVulkanState(d->vulkanState);
    tcannyDestroyCudaState(d->cudaState);
    tcannyDestroyCpuPool(d->cpuPool);
    vsapi->freeNode(d->node);
    delete d;
}

void VS_CC tcannyCreate(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) {
    auto data = std::make_unique<TCannyData>();
    int error = 0;
    try {
        data->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        data->vi = vsapi->getVideoInfo(data->node);
        if (!isConstantSupported(data->vi)) throw std::runtime_error("only constant 8-16 bit integer or 32-bit float input is supported");
        const int numPlanes = data->vi->format.numPlanes;
        const int sigmaHCount = vsapi->mapNumElements(in, "sigma");
        const int sigmaVCount = vsapi->mapNumElements(in, "sigma_v");
        if (sigmaHCount > numPlanes || sigmaVCount > numPlanes) throw std::runtime_error("too many sigma values");
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (plane < sigmaHCount) data->sigmaH[plane] = static_cast<float>(vsapi->mapGetFloatSaturated(in, "sigma", plane, nullptr));
            else if (plane == 0) data->sigmaH[plane] = 1.5f;
            else if (plane == 1) data->sigmaH[plane] = data->sigmaH[0] / (1 << data->vi->format.subSamplingW);
            else data->sigmaH[plane] = data->sigmaH[1];
            if (plane < sigmaVCount) data->sigmaV[plane] = static_cast<float>(vsapi->mapGetFloatSaturated(in, "sigma_v", plane, nullptr));
            else if (plane < sigmaHCount) data->sigmaV[plane] = data->sigmaH[plane];
            else if (plane == 0) data->sigmaV[plane] = 1.5f;
            else if (plane == 1) data->sigmaV[plane] = data->sigmaV[0] / (1 << data->vi->format.subSamplingH);
            else data->sigmaV[plane] = data->sigmaV[1];
            if (data->sigmaH[plane] < 0.0f || data->sigmaV[plane] < 0.0f) throw std::runtime_error("sigma must be non-negative");
            if (data->sigmaH[plane] > 0.0f) data->weightsH[plane] = gaussianWeights(data->sigmaH[plane], data->radiusH[plane]);
            if (data->sigmaV[plane] > 0.0f) data->weightsV[plane] = gaussianWeights(data->sigmaV[plane], data->radiusV[plane]);
        }
        data->t_h = static_cast<float>(vsapi->mapGetFloatSaturated(in, "t_h", 0, &error)); if (error) { error = 0; data->t_h = 8.0f; }
        data->t_l = static_cast<float>(vsapi->mapGetFloatSaturated(in, "t_l", 0, &error)); if (error) { error = 0; data->t_l = 1.0f; }
        data->mode = vsapi->mapGetIntSaturated(in, "mode", 0, &error); if (error) { error = 0; data->mode = 0; }
        data->op = vsapi->mapGetIntSaturated(in, "op", 0, &error); if (error) { error = 0; data->op = PREWITT; }
        data->scale = static_cast<float>(vsapi->mapGetFloatSaturated(in, "scale", 0, &error)); if (error) { error = 0; data->scale = 1.0f; }
        const int opt = vsapi->mapGetIntSaturated(in, "opt", 0, &error);
        if (!error && (opt < 0 || opt > 4)) throw std::runtime_error("opt must be 0, 1, 2, 3, or 4");
        error = 0;
        data->opt = opt;
        tcannySetupCpu(data.get());
        const int m = vsapi->mapNumElements(in, "planes");
        data->process.fill(m <= 0);
        for (int i = 0; i < m; ++i) {
            const int plane = vsapi->mapGetIntSaturated(in, "planes", i, nullptr);
            if (plane < 0 || plane >= numPlanes || data->process[plane]) throw std::runtime_error("invalid or duplicate plane index");
            data->process[plane] = true;
        }
        if (data->t_l >= data->t_h) throw std::runtime_error("t_h must be greater than t_l");
        if (data->mode < -1 || data->mode > 1) throw std::runtime_error("mode must be -1, 0, or 1");
        if (data->op < 0 || data->op > 6 || (data->op == KIRSCH && data->mode == 0)) throw std::runtime_error("invalid operator for mode");
        if (data->scale <= 0.0f) throw std::runtime_error("scale must be greater than zero");
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!data->process[plane]) continue;
            const int pw = data->vi->width >> (plane ? data->vi->format.subSamplingW : 0);
            const int ph = data->vi->height >> (plane ? data->vi->format.subSamplingH : 0);
            std::ostringstream msg;
            msg << "plane " << plane << " is " << pw << "x" << ph << ", which is too small: ";
            if (pw <= data->radiusH[plane])
                throw std::runtime_error(msg.str() + "width must exceed the horizontal blur radius (" + std::to_string(data->radiusH[plane]) + ")");
            if (ph <= data->radiusV[plane])
                throw std::runtime_error(msg.str() + "height must exceed the vertical blur radius (" + std::to_string(data->radiusV[plane]) + ")");
        }
        if (data->mode != -1) {
            // 3 for every operator: the GPU backends clamp instead of mirror below 3, and FDOG needs 3 anyway.
            const int minSize = 3;
            for (int plane = 0; plane < numPlanes; ++plane) {
                if (!data->process[plane]) continue;
                const int pw = data->vi->width >> (plane ? data->vi->format.subSamplingW : 0);
                const int ph = data->vi->height >> (plane ? data->vi->format.subSamplingH : 0);
                if (pw < minSize || ph < minSize)
                    throw std::runtime_error("plane " + std::to_string(plane) + " is " + std::to_string(pw) + "x" + std::to_string(ph) +
                                             ", but the edge operator needs at least " + std::to_string(minSize) + "x" + std::to_string(minSize));
            }
        }
#if VAPOURSYNTH_API_MINOR >= 3
        const bool api43 = vsapi->getAPIVersion() >= VS_MAKE_VERSION(4, 3);
#else
        [[maybe_unused]] constexpr bool api43 = false;
#endif
        bool gpuInput = false;
#if VAPOURSYNTH_API_MINOR >= 3
        if (api43)
            gpuInput = vsapi->getNodeResidency(data->node) == nrGPU;
#endif
        const char *backend = vsapi->mapGetData(in, "backend", 0, &error);
        if (error || !backend || std::string_view(backend) == "auto") {
            error = 0;
            data->backend = gpuInput ? TCannyBackend::Vulkan : TCannyBackend::Cpu;
        } else if (std::string_view(backend) == "cpu") data->backend = TCannyBackend::Cpu;
        else if (std::string_view(backend) == "vulkan") data->backend = TCannyBackend::Vulkan;
        else if (std::string_view(backend) == "cuda") data->backend = TCannyBackend::Cuda;
        else throw std::runtime_error("backend must be auto, cpu, vulkan, or cuda");
        if (data->backend != TCannyBackend::Cpu && !api43)
            throw std::runtime_error("GPU backends require VapourSynth API 4.3 (R80 or newer)");
        if (data->backend == TCannyBackend::Cuda) {
            if (!tcannyCudaCompiled()) throw std::runtime_error("backend=cuda requested but CUDA support was not compiled into this build");
#ifdef _WIN32
            // The Windows interop path (NT handles, declaring held imports with reserveGPUMemory) has never been built or run on a GPU.
            throw std::runtime_error("backend=cuda is not supported on Windows yet; use backend=vulkan");
#endif
        }
        if (data->backend == TCannyBackend::Cpu && gpuInput) throw std::runtime_error("backend=cpu requires a CPU node; call std.GPUDownload first");
        if (data->backend != TCannyBackend::Cpu && !gpuInput) throw std::runtime_error("GPU backend requires a GPU node; call std.GPUUpload first");
        data->gpuOutput = data->backend != TCannyBackend::Cpu;
        if (data->backend == TCannyBackend::Vulkan) {
            std::string gpuError;
            data->vulkanState = tcannyCreateVulkanState(core, vsapi, data.get(), gpuError);
            if (!data->vulkanState)
                throw std::runtime_error(gpuError.empty() ? "failed to initialize Vulkan compute state" : gpuError);
        }
        if (data->backend == TCannyBackend::Cuda) {
            std::string gpuError;
            data->cudaState = tcannyCreateCudaState(gpuError);
            if (!data->cudaState)
                throw std::runtime_error(gpuError.empty() ? "failed to initialize CUDA state" : gpuError);
        }
        data->peak = data->vi->format.sampleType == stInteger ? (1 << data->vi->format.bitsPerSample) - 1 : 1;
        const float thresholdScale = data->vi->format.sampleType == stInteger ? data->peak / 255.0f : 1.0f / 255.0f;
        data->t_h *= thresholdScale;
        data->t_l *= thresholdScale;
        VSFilterDependency dep{data->node, rpStrictSpatial};
#if VAPOURSYNTH_API_MINOR >= 3
        VSNode *node = api43
            ? vsapi->createVideoFilterEx2("TCanny", data->vi, tcannyGetFrame, tcannyFree, fmParallel,
                                          data->gpuOutput ? ffGPUOutput : 0, &dep, 1, data.get(), core)
            : vsapi->createVideoFilter2("TCanny", data->vi, tcannyGetFrame, tcannyFree, fmParallel,
                                        &dep, 1, data.get(), core);
#else
        VSNode *node = vsapi->createVideoFilter2("TCanny", data->vi, tcannyGetFrame, tcannyFree, fmParallel,
                                                 &dep, 1, data.get(), core);
#endif
        if (!node) throw std::runtime_error("failed to create filter");
        vsapi->mapConsumeNode(out, "clip", node, maAppend);
        data.release();
    } catch (const std::exception &e) {
        tcannyDestroyVulkanState(data->vulkanState);
        tcannyDestroyCudaState(data->cudaState);
        tcannyDestroyCpuPool(data->cpuPool);
        if (data->node) vsapi->freeNode(data->node);
        vsapi->mapSetError(out, ("TCanny: " + std::string(e.what())).c_str());
    }
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    const int apiVersion = vspapi->getAPIVersion() >= VS_MAKE_VERSION(4, 3) ? VAPOURSYNTH_API_VERSION : VS_MAKE_VERSION(4, 2);
    vspapi->configPlugin("com.mystdove.tcannyx", "tcannyx", "Build an edge map using Canny edge detection (CPU, Vulkan and CUDA backends)", VS_MAKE_VERSION(0, 0), apiVersion, 0, plugin);
    const char *clipType = vspapi->getAPIVersion() >= VS_MAKE_VERSION(4, 3) ? "vnode:all" : "vnode";
    const std::string args = std::string("clip:") + clipType + ";"
        "sigma:float[]:opt;"
        "sigma_v:float[]:opt;"
        "t_h:float:opt;"
        "t_l:float:opt;"
        "mode:int:opt;"
        "op:int:opt;"
        "scale:float:opt;"
        "opt:int:opt;"
        "planes:int[]:opt;"
        "backend:data:opt;";
    const std::string returns = std::string("clip:") + clipType + ";";
    vspapi->registerFunction("TCanny", args.c_str(), returns.c_str(), tcannyCreate, nullptr, plugin);
}

