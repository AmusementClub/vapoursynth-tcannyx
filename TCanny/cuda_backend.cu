#include "TCanny.h"
#include "hysteresis_schedule.h"
#include "prof.h"
#include <VSVulkan4.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
__device__ int mirror_coord(int value, int limit) {
    if (limit < 2) return 0;
    // Single reflection (x=-1 -> 1, x=limit -> limit-2); the final clamp only
    // guards callers that would otherwise reflect past the opposite edge.
    if (value < 0) value = -value;
    else if (value >= limit) value = 2 * limit - 2 - value;
    return max(0, min(limit - 1, value));
}

// Blur kernels take the CPU's own radius and normalised weights (weights has
// 2*radius+1 entries). V is applied first and H second, as in the CPU path.
// Samples are read as float (integers unscaled, like the CPU path) and written back by storeSample: float output is
// raw (mode 1 clamps to 0..1), integer output is rounded half up and clamped to the sample peak (CPU: min(int(v + 0.5f), peak)).
__device__ __forceinline__ void storeSample(float *p, float v, int mode, int) { *p = mode == 1 ? fminf(fmaxf(v, 0.0f), 1.0f) : v; }
__device__ __forceinline__ void storeSample(unsigned char *p, float v, int, int peak) { *p = static_cast<unsigned char>(min(static_cast<int>(v + 0.5f), peak)); }
__device__ __forceinline__ void storeSample(unsigned short *p, float v, int, int peak) { *p = static_cast<unsigned short>(min(static_cast<int>(v + 0.5f), peak)); }
__device__ __forceinline__ void storeConstant(float *p, bool on, int) { *p = on ? 1.0f : 0.0f; }
__device__ __forceinline__ void storeConstant(unsigned char *p, bool on, int peak) { *p = on ? static_cast<unsigned char>(peak) : 0; }
__device__ __forceinline__ void storeConstant(unsigned short *p, bool on, int peak) { *p = on ? static_cast<unsigned short>(peak) : 0; }

template <typename S>
__global__ void tcanny_blur_h(const S *src, float *dst, int width, int height, int srcStride, int dstStride, int radius, const float *weights) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx)
        sum += static_cast<float>(src[static_cast<size_t>(y) * srcStride + mirror_coord(x + dx, width)]) * weights[dx + radius];
    dst[static_cast<size_t>(y) * dstStride + x] = sum;
}

template <typename S>
__global__ void tcanny_blur_v(const S *src, float *dst, int width, int height, int srcStride, int dstStride, int radius, const float *weights) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
        sum += static_cast<float>(src[static_cast<size_t>(mirror_coord(y + dy, height)) * srcStride + x]) * weights[dy + radius];
    dst[static_cast<size_t>(y) * dstStride + x] = sum;
}

// Modes -1 and 1 write the result to dst. Mode 0 evaluates the gradient once per pixel: the magnitude goes to `magnitude`
// (row stride magnitudeStride) and the direction bin (0..3, as a float) to dst; tcanny_nms resolves them next.
template <typename T>
__global__ void tcanny_kernel(const T *src, T *dst, float *state, const float *blur, float *magnitude, int width, int height, int srcStride, int dstStride,
                              int stateStride, int blurStride, int magnitudeStride, int mode, int op, float scale, int peak) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const auto raw = [&](int sx, int sy) { sx = mirror_coord(sx, width); sy = mirror_coord(sy, height); return static_cast<float>(src[sy * srcStride + sx]); };
    const auto sample = [&](int sx, int sy) {
        sx = mirror_coord(sx, width); sy = mirror_coord(sy, height);
        return blur ? blur[sy * blurStride + sx] : raw(sx, sy);
    };
    const float c1 = sample(x - 1, y - 1), c2 = sample(x, y - 1), c3 = sample(x + 1, y - 1);
    const float c4 = sample(x - 1, y), c5 = sample(x, y), c6 = sample(x + 1, y);
    const float c7 = sample(x - 1, y + 1), c8 = sample(x, y + 1), c9 = sample(x + 1, y + 1);
    if (mode == -1) { storeSample(&dst[y * dstStride + x], c5, mode, peak); return; }
    float gx, gy;
    if (op == TRITICAL) { gx = c6 - c4; gy = c2 - c8; }
    else if (op == SOBEL) { gx = c3 + 2.0f * c6 + c9 - c1 - 2.0f * c4 - c7; gy = c1 + 2.0f * c2 + c3 - c7 - 2.0f * c8 - c9; }
    else if (op == SCHARR) { gx = 3.0f*c3 + 10.0f*c6 + 3.0f*c9 - 3.0f*c1 - 10.0f*c4 - 3.0f*c7; gy = 3.0f*c1 + 10.0f*c2 + 3.0f*c3 - 3.0f*c7 - 10.0f*c8 - 3.0f*c9; }
    else if (op == KROON) { gx = 17.0f*c3 + 61.0f*c6 + 17.0f*c9 - 17.0f*c1 - 61.0f*c4 - 17.0f*c7; gy = 17.0f*c1 + 61.0f*c2 + 17.0f*c3 - 17.0f*c7 - 61.0f*c8 - 17.0f*c9; }
    else if (op == FDOG) {
        const float a1=sample(x-2,y-2), a2=sample(x-1,y-2), a3=sample(x,y-2), a4=sample(x+1,y-2), a5=sample(x+2,y-2);
        const float a6=sample(x-2,y-1), a7=sample(x-1,y-1), a8=sample(x,y-1), a9=sample(x+1,y-1), a10=sample(x+2,y-1);
        const float a11=sample(x-2,y), a12=sample(x-1,y), a14=sample(x+1,y), a15=sample(x+2,y);
        const float a16=sample(x-2,y+1), a17=sample(x-1,y+1), a18=sample(x,y+1), a19=sample(x+1,y+1), a20=sample(x+2,y+1);
        const float a21=sample(x-2,y+2), a22=sample(x-1,y+2), a23=sample(x,y+2), a24=sample(x+1,y+2), a25=sample(x+2,y+2);
        gx = a5 + 2.0f*a10 + 3.0f*a15 + 2.0f*a20 + a25 + a4 + 2.0f*a9 + 3.0f*a14 + 2.0f*a19 + a24 - a2 - 2.0f*a7 - 3.0f*a12 - 2.0f*a17 - a22 - a1 - 2.0f*a6 - 3.0f*a11 - 2.0f*a16 - a21;
        gy = a1 + 2.0f*a2 + 3.0f*a3 + 2.0f*a4 + a5 + a6 + 2.0f*a7 + 3.0f*a8 + 2.0f*a9 + a10 - a16 - 2.0f*a17 - 3.0f*a18 - 2.0f*a19 - a20 - a21 - 2.0f*a22 - 3.0f*a23 - 2.0f*a24 - a25;
    }
    else if (op == KIRSCH) {
        const float g1 = 5.0f*c1 + 5.0f*c2 + 5.0f*c3 - 3.0f*c4 - 3.0f*c6 - 3.0f*c7 - 3.0f*c8 - 3.0f*c9;
        const float g2 = 5.0f*c1 + 5.0f*c2 - 3.0f*c3 + 5.0f*c4 - 3.0f*c6 - 3.0f*c7 - 3.0f*c8 - 3.0f*c9;
        const float g3 = 5.0f*c1 - 3.0f*c2 - 3.0f*c3 + 5.0f*c4 - 3.0f*c6 + 5.0f*c7 - 3.0f*c8 - 3.0f*c9;
        const float g4 = -3.0f*c1 - 3.0f*c2 - 3.0f*c3 + 5.0f*c4 - 3.0f*c6 + 5.0f*c7 + 5.0f*c8 - 3.0f*c9;
        const float g5 = -3.0f*c1 - 3.0f*c2 - 3.0f*c3 - 3.0f*c4 - 3.0f*c6 + 5.0f*c7 + 5.0f*c8 + 5.0f*c9;
        const float g6 = -3.0f*c1 - 3.0f*c2 - 3.0f*c3 - 3.0f*c4 + 5.0f*c6 - 3.0f*c7 + 5.0f*c8 + 5.0f*c9;
        const float g7 = -3.0f*c1 - 3.0f*c2 + 5.0f*c3 - 3.0f*c4 + 5.0f*c6 - 3.0f*c7 - 3.0f*c8 + 5.0f*c9;
        const float g8 = -3.0f*c1 + 5.0f*c2 + 5.0f*c3 - 3.0f*c4 + 5.0f*c6 - 3.0f*c7 - 3.0f*c8 - 3.0f*c9;
        gx = max(max(max(fabsf(g1),fabsf(g2)),max(fabsf(g3),fabsf(g4))),max(max(fabsf(g5),fabsf(g6)),max(fabsf(g7),fabsf(g8))));
        gy = 0.0f;
    }
    else { gx = (c3 + c6 + c9 - c1 - c4 - c7) * 0.5f; gy = (c1 + c2 + c3 - c7 - c8 - c9) * 0.5f; }
    // Scale the components first, as the CPU reference and the Vulkan shader do (the other order rounds differently).
    float value;
    if (op == KIRSCH) value = fabsf(gx) * scale;
    else { gx *= scale; gy *= scale; value = sqrtf(gx * gx + gy * gy); }
    if (mode == 0) {
        float angle = atan2f(gy, gx);
        if (angle < 0.0f) angle += 3.14159265358979323846f;
        const int direction = static_cast<int>(angle * 4.0f * 0.318309886183790671538f + 0.5f) & 3;
        magnitude[static_cast<size_t>(y) * magnitudeStride + x] = value;
        state[y * stateStride + x] = static_cast<float>(direction);
        return;
    }
    storeSample(&dst[y * dstStride + x], value, mode, peak);
}

// Non-maximum suppression on the stored magnitudes: dst holds the direction bin on entry and the state (1.0 strong,
// 0.5 weak, 0.0 none) on exit. A pixel reads and writes only its own dst element.
__global__ void tcanny_nms(float *dst, const float *magnitude, int width, int height, int dstStride, int magnitudeStride, float highThreshold,
                           float lowThreshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int direction = static_cast<int>(dst[y * dstStride + x]);
    const int dx = direction == 0 ? 1 : direction == 1 ? 1 : direction == 2 ? 0 : -1;
    const int dy = direction == 0 ? 0 : direction == 1 ? -1 : direction == 2 ? -1 : -1;
    const float value = magnitude[static_cast<size_t>(y) * magnitudeStride + x];
    const float n1 = magnitude[static_cast<size_t>(mirror_coord(y + dy, height)) * magnitudeStride + mirror_coord(x + dx, width)];
    const float n2 = magnitude[static_cast<size_t>(mirror_coord(y - dy, height)) * magnitudeStride + mirror_coord(x - dx, width)];
    float state = 0.0f;
    if (value >= n1 && value >= n2 && value >= highThreshold) state = 1.0f;
    else if (value >= n1 && value >= n2 && value >= lowThreshold) state = 0.5f;
    dst[y * dstStride + x] = state;
}

// Hysteresis on bitplanes, mirroring the Vulkan shaders (TCanny/shaders/tcanny_pack.comp.glsl,
// tcanny_hysteresis_tile.comp.glsl and tcanny_hysteresis.comp.glsl, the last one only the final write); keep the two textually close so that a fix in one ports to the other.
// In mode 0 tcanny_kernel leaves a per-pixel state in a float plane (the destination itself for float input, the slot's
// `stateBuf` for integer input): 1.0 strong, 0.5 weak, 0.0 none.

// Packs the states into two bitplanes, one bit per pixel (bit i of word wx is pixel x = wx * 32 + i): `cand` (any
// state) and `strong`. One warp per word: lane i looks at pixel wx * 32 + i and the ballots are the words. Every lane
// takes part in the ballots (pixels outside the plane count as 0), so bits at x >= width stay 0.
__global__ void tcanny_pack(const float *image, int width, int height, int stride, unsigned *cand, unsigned *strong, int wordsPerRow) {
    const int wx = blockIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = wx * 32 + threadIdx.x;
    const float value = (x < width && y < height) ? image[static_cast<size_t>(y) * stride + x] : 0.0f;
    const unsigned c = __ballot_sync(0xffffffffu, value != 0.0f);
    const unsigned s = __ballot_sync(0xffffffffu, value >= 1.0f);
    if (threadIdx.x == 0 && y < height) {
        cand[static_cast<size_t>(y) * wordsPerRow + wx] = c;
        strong[static_cast<size_t>(y) * wordsPerRow + wx] = s;
    }
}

// Promotion pass (mirror of tcanny_hysteresis_tile.comp.glsl). Promote every candidate bit with a strong bit among its 8
// neighbours (rows y-1, y, y+1; neighbouring words supply bit -1 and bit 32) and fill along the row inside the word until
// stable; out-of-range words and rows read as 0. One thread owns one word column of kTileRows consecutive rows: it loads the tile and a one-row / one-word halo, sweeps down and up in Gauss-Seidel order until nothing
// changes or kMaxPairs pairs are done, and writes back the words it changed. Halo words are read once per pass and may be
// stale, which only delays a promotion. Raises *changed to passIndex + 1 when any word changed, the cap included.
constexpr int kTileRows = 8;
constexpr int kMaxPairs = 2;
__device__ __forceinline__ unsigned spreadBits(unsigned n) { return n | (n << 1) | (n >> 1); }
__global__ void tcanny_hysteresis_tile(const unsigned *cand, volatile unsigned *strong, int height, int wordsPerRow, int *changed, int passIndex) {
    const int wx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = blockIdx.y * kTileRows;
    if (wx >= wordsPerRow || y0 >= height) return;
    unsigned s[kTileRows + 2], e[kTileRows + 2], c[kTileRows + 2];
#pragma unroll
    for (int i = 0; i < kTileRows + 2; ++i) {
        const int row = y0 + i - 1;
        unsigned sv = 0u, ev = 0u, cv = 0u;
        if (row >= 0 && row < height) {
            const size_t base = static_cast<size_t>(row) * wordsPerRow;
            sv = strong[base + wx];
            const unsigned l = wx > 0 ? strong[base + wx - 1] : 0u;
            const unsigned r = wx + 1 < wordsPerRow ? strong[base + wx + 1] : 0u;
            ev = (l >> 31) | (r << 31);
            if (i >= 1 && i <= kTileRows) cv = cand[base + wx];
        }
        s[i] = sv; e[i] = ev; c[i] = cv;
    }
    unsigned dirty = 0u;
    unsigned progress = 0u;
    for (int pair = 0; pair < kMaxPairs; ++pair) {
        const unsigned before = progress;
#pragma unroll
        for (int k = 0; k < 2 * kTileRows; ++k) {
            const int i = k < kTileRows ? k + 1 : 2 * kTileRows - k;
            if (c[i] != s[i]) {
                const unsigned around = spreadBits(s[i - 1]) | e[i - 1] | spreadBits(s[i]) | e[i] | spreadBits(s[i + 1]) | e[i + 1];
                unsigned t = s[i] | (c[i] & around);
                for (;;) {
                    const unsigned u = t;
                    t |= c[i] & ((t << 1) | (t >> 1));
                    if (t == u) break;
                }
                if (t != s[i]) { s[i] = t; dirty |= 1u << (i - 1); ++progress; }
            }
        }
        if (progress == before) break;
    }
    if (dirty == 0u) return;
#pragma unroll
    for (int i = 1; i <= kTileRows; ++i)
        if (dirty & (1u << (i - 1))) strong[static_cast<size_t>(y0 + i - 1) * wordsPerRow + wx] = s[i];
    atomicMax(changed, passIndex + 1);
}

// Writes the destination plane from the strong bits: the sample peak (1.0 for float) for strong, 0 for everything else.
template <typename T>
__global__ void tcanny_finalize(T *image, const unsigned *strong, int width, int height, int stride, int wordsPerRow, int peak) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const unsigned bit = (strong[static_cast<size_t>(y) * wordsPerRow + (x >> 5)] >> (x & 31)) & 1u;
    storeConstant(&image[static_cast<size_t>(y) * stride + x], bit != 0u, peak);
}

// Timing accumulators (empty without -Dprofiling). Per-process, summed over worker threads.
struct CudaProf {
    tcprof::Acc total, wait, select, makeCur, newFrame, plane, importPlanes, launches, hysteresis, finalSync, publish, slotWait, init;
    std::atomic<uint64_t> frames{0};
};
CudaProf g_prof;
inline void g_prof_frames_add() {
#ifdef TCANNY_PROF
    g_prof.frames.fetch_add(1);
#endif
}

bool cudaCheck(cudaError_t status, const char *what, std::string &error) {
    if (status == cudaSuccess) return true;
    error = std::string(what) + ": " + cudaGetErrorString(status);
    return false;
}

// ---- RAII helpers: every exit path releases exactly what it owns ----

// Close an exported OS handle that we still own.
void closeNative(intptr_t handle) {
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
    close(static_cast<int>(handle));
#endif
}

// Ownership rule after handing an exported handle to a CUDA import:
//  - Windows: NT handles are never consumed by the import, always close.
//  - POSIX: a successful import consumes the fd; close only if the import failed.
void finishImportedHandle(intptr_t handle, bool importSucceeded) {
#ifdef _WIN32
    (void)importSucceeded;
    closeNative(handle);
#else
    if (!importSucceeded) closeNative(handle);
#endif
}

// An imported allocation with a buffer mapped over all of it; planes are addressed by their offset. The CUDA
// documentation requires the mapped pointer to be freed with cudaFree before the external memory is destroyed;
// destroying only the handle leaks the mapping (about 40 KB of device-side bookkeeping each, which showed up as steady
// VRAM growth over thousands of frames). The last reference releases both, on whichever thread holds it, with the
// instance's device current.
struct ImportedMemory {
    cudaExternalMemory_t handle = nullptr;
    void *base = nullptr;
    int device = -1; // the device the import lives on; made current while releasing (the last holder may be any thread)
    ImportedMemory() = default;
    ImportedMemory(const ImportedMemory &) = delete;
    ImportedMemory &operator=(const ImportedMemory &) = delete;
    ~ImportedMemory() {
        int previous = -1;
        const bool switched = device >= 0 && cudaGetDevice(&previous) == cudaSuccess && previous != device && cudaSetDevice(device) == cudaSuccess;
        if (base) cudaFree(base);
        if (handle) cudaDestroyExternalMemory(handle);
        if (switched) cudaSetDevice(previous);
    }
};

struct FrameGuard {
    const VSAPI *api;
    VSFrame *frame;
    ~FrameGuard() { if (frame) api->freeFrame(frame); }
    VSFrame *release() { VSFrame *f = frame; frame = nullptr; return f; }
};

// Maps a Vulkan device UUID to a CUDA device index. Enumerating every device's properties costs about 0.3 ms, so
// each UUID is resolved once per process.
bool resolveDevice(const uint8_t *uuid, int &device, std::string &error) {
    static std::mutex mutex;
    static std::vector<std::pair<std::array<uint8_t, VK_UUID_SIZE>, int>> known;
    std::array<uint8_t, VK_UUID_SIZE> key;
    std::memcpy(key.data(), uuid, VK_UUID_SIZE);
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &entry : known)
        if (entry.first == key) { device = entry.second; return true; }
    int count = 0;
    if (!cudaCheck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", error)) return false;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (!cudaCheck(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties", error)) return false;
        if (std::memcmp(prop.uuid.bytes, uuid, VK_UUID_SIZE) == 0) { known.emplace_back(key, i); device = i; return true; }
    }
    error = "no CUDA device matches VapourSynth's Vulkan device UUID";
    return false;
}

// Which CUDA device matches this core's Vulkan device. Asking the core costs about 0.16 ms (and contends between
// threads), so the answer is kept in the filter instance, which belongs to exactly one core.
bool resolveCoreDevice(const VSVULKANAPI *vk, VSCore *core, int &device, std::string &error) {
    char message[512]{};
    VSVulkanCoreInfo info{};
    if (vk->getVulkanCoreInfo(core, &info, message, sizeof(message)) != 0) { error = std::string("Vulkan device info failed: ") + message; return false; }
    return resolveDevice(info.deviceUUID, device, error);
}

// cudaSetDevice is per host thread and VapourSynth worker threads are not the thread that last called it, so the
// current device is checked for every frame and only changed when it differs.
bool makeCurrent(int device, std::string &error) {
    int current = -1;
    if (cudaGetDevice(&current) == cudaSuccess && current == device) return true;
    return cudaCheck(cudaSetDevice(device), "cudaSetDevice", error);
}

// Takes ownership of exported.handle: it is closed here whenever CUDA did not
// consume it (see finishImportedHandle), including every early failure. Maps the whole allocation once.
std::shared_ptr<ImportedMemory> importMemory(const VSVulkanExportedMemory &exported, int device, std::string &error) {
    cudaExternalMemoryHandleDesc desc{};
#ifdef _WIN32
    if (exported.handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT) { closeNative(exported.handle); error = "unsupported Windows Vulkan external-memory handle type"; return nullptr; }
    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = reinterpret_cast<void *>(exported.handle);
#else
    if (exported.handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) { closeNative(exported.handle); error = "unsupported POSIX Vulkan external-memory handle type"; return nullptr; }
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = static_cast<int>(exported.handle);
#endif
    desc.size = exported.memorySize;
    auto memory = std::make_shared<ImportedMemory>();
    memory->device = device;
    const cudaError_t status = cudaImportExternalMemory(&memory->handle, &desc);
    finishImportedHandle(exported.handle, status == cudaSuccess);
    if (!cudaCheck(status, "cudaImportExternalMemory", error)) { memory->handle = nullptr; return nullptr; }
    cudaExternalMemoryBufferDesc bufferDesc{};
    bufferDesc.offset = 0;
    bufferDesc.size = exported.memorySize;
    if (!cudaCheck(cudaExternalMemoryGetMappedBuffer(&memory->base, memory->handle, &bufferDesc), "cudaExternalMemoryGetMappedBuffer", error)) { memory->base = nullptr; return nullptr; }
    return memory;
}

// Imports keyed by the core's memoryId (stable for an allocation's lifetime, never reused; every export returns a new
// handle, so the handle is never the key). A cached import keeps a block alive after the core has freed it, and the
// core no longer counts it, so the cache is bounded two ways: least recently used entries go beyond kMaxImports, and
// entries unused for kIdleSeconds are dropped by a sweep that runs on lookups at most once per kSweepSeconds, hits
// included. (A process that stops asking cannot sweep: no timer thread.) A dropped entry is only released when the
// last frame still using it lets go (shared ownership). One cache serves every filter instance on one core and device
// (see sharedImportCache), so the same block is imported once.
class ImportCache {
public:
    static constexpr size_t kMaxImports = 16;
    static constexpr double kIdleSeconds = 5.0;
    static constexpr double kSweepSeconds = 1.0;
    using Clock = std::chrono::steady_clock;

    explicit ImportCache(int device) : device_(device) {}

    // Consumes exported.handle on every path. Returns null with `error` set on failure.
    std::shared_ptr<ImportedMemory> acquire(const VSVulkanExportedMemory &exported, std::string &error) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = Clock::now();
        if (seconds(now - lastSweep_) >= kSweepSeconds) {
            lastSweep_ = now;
            for (auto it = entries_.begin(); it != entries_.end();) {
                if (seconds(now - it->second.lastUse) > kIdleSeconds) it = entries_.erase(it);
                else ++it;
            }
        }
        const auto found = entries_.find(exported.memoryId);
        if (found != entries_.end()) {
            found->second.lastUse = now;
            closeNative(exported.handle); // surplus: this export's handle was not consumed by an import
            return found->second.memory;
        }
        auto memory = importMemory(exported, device_, error);
        if (!memory) return nullptr;
        while (entries_.size() >= kMaxImports) {
            auto oldest = entries_.begin();
            for (auto it = entries_.begin(); it != entries_.end(); ++it)
                if (it->second.lastUse < oldest->second.lastUse) oldest = it;
            entries_.erase(oldest);
        }
        entries_.emplace(exported.memoryId, Entry{memory, now});
        peak_ = std::max(peak_, entries_.size());
        return memory;
    }

    size_t peakEntries() {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

private:
    static double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
    struct Entry {
        std::shared_ptr<ImportedMemory> memory;
        Clock::time_point lastUse;
    };
    int device_;
    std::mutex mutex_;
    std::unordered_map<uint64_t, Entry> entries_;
    Clock::time_point lastSweep_ = Clock::now();
    size_t peak_ = 0;
};

// One import cache per (core, CUDA device), shared by every filter instance that uses it and released with the last
// of them. Keyed by the core as well because memoryId uniqueness is only promised within a core.
std::shared_ptr<ImportCache> sharedImportCache(VSCore *core, int device) {
    static std::mutex mutex;
    static std::map<std::pair<VSCore *, int>, std::weak_ptr<ImportCache>> caches;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = caches.begin(); it != caches.end();) {
        if (it->second.expired()) it = caches.erase(it);
        else ++it;
    }
    auto &slot = caches[{core, device}];
    if (auto existing = slot.lock()) return existing;
    auto fresh = std::make_shared<ImportCache>(device);
    slot = fresh;
    return fresh;
}

#ifdef TCANNY_PROF
// TCANNY_PROF_BATCH=k forces every hysteresis batch to k passes (exact-depth counting runs only); 0 = scheduled.
uint32_t forcedBatch() {
    static const uint32_t forced = [] {
        const char *v = std::getenv("TCANNY_PROF_BATCH");
        return v ? static_cast<uint32_t>(std::max(0, std::atoi(v))) : 0u;
    }();
    return forced;
}
#else
constexpr uint32_t forcedBatch() { return 0; }
#endif

// ---- per-instance state: device weights and a gated pool of slots ----

// A slot is everything one in-flight frame needs on the device: its stream, the two blur temporaries (sized for the
// largest processed plane and shared by all planes of the frame, which run back to back on the one stream), the
// device-side changed flag of the hysteresis loop and the pinned host word it is read back into.
struct CudaSlot {
    cudaStream_t stream = nullptr;
    void *bufferV = nullptr, *bufferH = nullptr;
    void *stateBuf = nullptr; // float state plane of mode 0 for integer formats (the destination cannot hold it)
    int *changed = nullptr;
    int *hostFlag = nullptr; // pinned
    unsigned *cand = nullptr, *strong = nullptr; // hysteresis bitplanes (mode 0)
    bool ready = false;

    // Best effort, never throws; used for teardown and for a slot whose stream reported an error.
    void release() noexcept {
        if (stream) { cudaStreamSynchronize(stream); }
        if (hostFlag) cudaFreeHost(hostFlag);
        if (changed) cudaFree(changed);
        if (strong) cudaFree(strong);
        if (cand) cudaFree(cand);
        if (stateBuf) cudaFree(stateBuf);
        if (bufferH) cudaFree(bufferH);
        if (bufferV) cudaFree(bufferV);
        if (stream) cudaStreamDestroy(stream);
        stream = nullptr; bufferV = bufferH = stateBuf = nullptr; changed = nullptr; hostFlag = nullptr; cand = strong = nullptr; ready = false;
    }
};

constexpr size_t kDefaultSlots = 4; // throughput peaks at about 4 concurrent frames; more only contend in the driver

struct CudaState {
    // Pool.
    std::mutex poolMutex;
    std::condition_variable poolCv;
    std::vector<std::unique_ptr<CudaSlot>> slots;
    std::vector<CudaSlot *> freeSlots;
    size_t maxSlots = kDefaultSlots;
    // Lazy device-side initialisation (needs the right CUDA device current, so it happens on a frame).
    std::mutex initMutex;
    bool initialized = false;
    int device = -1;
    float *weights = nullptr;
    std::array<size_t, 3> offsetV{}, offsetH{}; // in floats, inside `weights`
    std::atomic<int> resolvedDevice{-1};
    size_t maxWidth = 0, maxHeight = 0;
    bool needBits = false;
    bool needState = false; // mode 0 on an integer format
    bool needV = false, needH = false; // which blur temporaries any processed plane touches
    std::atomic<size_t> pitchBytes{0};
    std::shared_ptr<ImportCache> imports; // set by ensureInit, shared with the other instances on this core and device
    HysteresisDepthHint depthHint;

    ~CudaState() {
        if (device >= 0) cudaSetDevice(device);
        for (auto &slot : slots) slot->release(); // drains every stream before the imports go
        imports.reset();
        if (weights) cudaFree(weights);
    }
};

// Uploads the Gaussian weights of every processed plane once and records the temporaries' geometry.
bool ensureInit(CudaState &state, const TCannyData *data, VSCore *core, int device, std::string &error) {
    std::lock_guard<std::mutex> lock(state.initMutex);
    if (state.initialized) return true;
    std::vector<float> host;
    bool anyV = false, anyH = false;
    for (int plane = 0; plane < data->vi->format.numPlanes; ++plane) {
        if (!data->process[plane]) continue;
        const int radiusH = data->radiusH[plane], radiusV = data->radiusV[plane];
        if (radiusV > 0 && data->weightsV[plane].size() != size_t(2 * radiusV + 1)) { error = "inconsistent Gaussian weights"; return false; }
        if (radiusH > 0 && data->weightsH[plane].size() != size_t(2 * radiusH + 1)) { error = "inconsistent Gaussian weights"; return false; }
        state.offsetV[plane] = host.size();
        if (radiusV > 0) host.insert(host.end(), data->weightsV[plane].begin(), data->weightsV[plane].end());
        state.offsetH[plane] = host.size();
        if (radiusH > 0) host.insert(host.end(), data->weightsH[plane].begin(), data->weightsH[plane].end());
        const size_t w = size_t(data->vi->width) >> (plane ? data->vi->format.subSamplingW : 0);
        const size_t h = size_t(data->vi->height) >> (plane ? data->vi->format.subSamplingH : 0);
        anyV = anyV || radiusV > 0;
        anyH = anyH || radiusH > 0;
        state.maxWidth = std::max(state.maxWidth, w);
        state.maxHeight = std::max(state.maxHeight, h);
    }
    if (!host.empty()) {
        float *device_weights = nullptr;
        if (!cudaCheck(cudaMalloc(&device_weights, host.size() * sizeof(float)), "cudaMalloc(weights)", error)) return false;
        if (!cudaCheck(cudaMemcpy(device_weights, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), "weights upload", error)) {
            cudaFree(device_weights);
            return false;
        }
        state.weights = device_weights;
    }
    state.needBits = data->mode == 0;
    state.needState = data->mode == 0 && data->vi->format.sampleType == stInteger;
    // Mode 0 keeps the magnitudes in the temporary the blur did not end in: always the horizontal one when nothing is
    // blurred, otherwise both are in play. Modes +-1 only need the temporaries a blur writes.
    state.needV = anyV || (data->mode == 0 && anyH);
    state.needH = anyH || data->mode == 0;
    state.imports = sharedImportCache(core, device);
    state.device = device;
    state.initialized = true;
    return true;
}

bool initSlot(CudaState &state, CudaSlot &slot, std::string &error) {
    slot.release();
    if (!cudaCheck(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking), "cudaStreamCreate", error)) { slot.stream = nullptr; return false; }
    if (state.maxWidth && state.maxHeight) {
        size_t pitchV = 0, pitchH = 0;
        if (state.needV && !cudaCheck(cudaMallocPitch(&slot.bufferV, &pitchV, state.maxWidth * sizeof(float), state.maxHeight), "cudaMallocPitch(vertical)", error)) { slot.bufferV = nullptr; slot.release(); return false; }
        if (state.needH && !cudaCheck(cudaMallocPitch(&slot.bufferH, &pitchH, state.maxWidth * sizeof(float), state.maxHeight), "cudaMallocPitch(horizontal)", error)) { slot.bufferH = nullptr; slot.release(); return false; }
        size_t pitchS = 0;
        if (state.needState && !cudaCheck(cudaMallocPitch(&slot.stateBuf, &pitchS, state.maxWidth * sizeof(float), state.maxHeight), "cudaMallocPitch(state)", error)) { slot.stateBuf = nullptr; slot.release(); return false; }
        if (state.needV && state.needH && pitchV != pitchH) { error = "inconsistent pitches"; slot.release(); return false; }
        state.pitchBytes.store(state.needV ? pitchV : state.needH ? pitchH : pitchS); // identical for every slot: same call, same arguments
    }
    if (state.needBits) {
        const size_t words = ((state.maxWidth + 31) / 32) * state.maxHeight;
        if (!cudaCheck(cudaMalloc(reinterpret_cast<void **>(&slot.cand), words * sizeof(unsigned)), "cudaMalloc(candidate bits)", error)) { slot.cand = nullptr; slot.release(); return false; }
        if (!cudaCheck(cudaMalloc(reinterpret_cast<void **>(&slot.strong), words * sizeof(unsigned)), "cudaMalloc(strong bits)", error)) { slot.strong = nullptr; slot.release(); return false; }
    }
    if (!cudaCheck(cudaMalloc(reinterpret_cast<void **>(&slot.changed), sizeof(int)), "cudaMalloc(changed)", error)) { slot.changed = nullptr; slot.release(); return false; }
    if (!cudaCheck(cudaHostAlloc(reinterpret_cast<void **>(&slot.hostFlag), sizeof(int), cudaHostAllocDefault), "cudaHostAlloc(flag)", error)) { slot.hostFlag = nullptr; slot.release(); return false; }
    slot.ready = true;
    return true;
}

// Holds one slot for the duration of a frame. On release the slot's stream is drained; a slot whose stream reports an
// error is destroyed and rebuilt by the next frame that takes it, so one failed frame cannot poison later ones.
class SlotLease {
public:
    SlotLease(CudaState &state, std::string &error) : state_(state) {
        std::unique_lock<std::mutex> lock(state_.poolMutex);
        tcprof::Timer waitTimer(g_prof.slotWait);
        for (;;) {
            if (!state_.freeSlots.empty()) {
                slot_ = state_.freeSlots.back();
                state_.freeSlots.pop_back();
                break;
            }
            if (state_.slots.size() < state_.maxSlots) {
                state_.slots.push_back(std::make_unique<CudaSlot>());
                slot_ = state_.slots.back().get();
                break;
            }
            state_.poolCv.wait(lock);
        }
        lock.unlock();
        waitTimer.stop();
        if (!slot_->ready) {
            tcprof::Timer t(g_prof.init);
            if (!initSlot(state_, *slot_, error)) giveBack();
        }
    }
    ~SlotLease() { giveBack(); }
    SlotLease(const SlotLease &) = delete;
    SlotLease &operator=(const SlotLease &) = delete;
    CudaSlot *get() const noexcept { return slot_; }

private:
    void giveBack() noexcept {
        if (!slot_) return;
        if (slot_->stream && cudaStreamSynchronize(slot_->stream) != cudaSuccess) slot_->release();
        {
            std::lock_guard<std::mutex> lock(state_.poolMutex);
            state_.freeSlots.push_back(slot_);
        }
        state_.poolCv.notify_one();
        slot_ = nullptr;
    }
    CudaState &state_;
    CudaSlot *slot_ = nullptr;
};

// Memory imported for one frame; the references are held until the frame's stream has been drained.
using KeepAlive = std::vector<std::shared_ptr<ImportedMemory>>;

// Exports one plane and resolves it to a device pointer inside a cached import. The exported handle is consumed by
// ImportCache::acquire on every path, so a failure at any point leaves no handle open.
bool importPlane(const VSVULKANAPI *vk, CudaState &state, const VSFrame *frame, int plane, const char *what, KeepAlive &keep, void *&pointer,
                 std::string &error) {
    char message[512]{};
    VSVulkanExportedMemory exported{};
    if (vk->exportGPUPlane(frame, plane, &exported, message, sizeof(message)) != 0) { error = std::string("exportGPUPlane(") + what + ") failed: " + message; return false; }
    auto memory = state.imports->acquire(exported, error);
    if (!memory) return false;
    pointer = static_cast<char *>(memory->base) + exported.offset;
    keep.push_back(std::move(memory));
    return true;
}

bool importPlanePair(const VSVULKANAPI *vk, CudaState &state, const VSFrame *src, VSFrame *dst, int plane, KeepAlive &keep, void *&input,
                     void *&output, std::string &error) {
    tcprof::Timer t(g_prof.importPlanes);
    return importPlane(vk, state, src, plane, "src", keep, input, error) && importPlane(vk, state, dst, plane, "dst", keep, output, error);
}

// An unprocessed plane is copied into the (always freshly allocated) output
// frame: sharing the input plane is not allowed, since inputs taken for a
// foreign API never go back to the core.
bool copyPlane(const VSVULKANAPI *vk, const VSAPI *vsapi, CudaState &state, CudaSlot &slot, KeepAlive &keep, const VSFrame *src, VSFrame *dst, int plane,
               std::string &error) {
    void *input = nullptr, *output = nullptr;
    if (!importPlanePair(vk, state, src, dst, plane, keep, input, output, error)) return false;
    const size_t rowBytes = size_t(vsapi->getFrameWidth(src, plane)) * vsapi->getVideoFrameFormat(src)->bytesPerSample;
    return cudaCheck(cudaMemcpy2DAsync(output, vsapi->getStride(dst, plane), input, vsapi->getStride(src, plane), rowBytes,
                                       vsapi->getFrameHeight(src, plane), cudaMemcpyDeviceToDevice, slot.stream), "cudaMemcpy2DAsync(copy plane)", error);
}

// Enqueues one plane on the slot's stream. Planes of a frame run back to back on that stream (so they may share the
// slot's temporaries); the caller keeps the imports alive in `keep` and drains the stream once, after the last plane,
// except where this plane needs the host (the hysteresis flag).
template <typename T>
bool processPlaneAs(const VSVULKANAPI *vk, const VSAPI *vsapi, CudaState &state, CudaSlot &slot, KeepAlive &keep, const TCannyData *data,
                    const VSFrame *src, VSFrame *dst, int plane, std::string &error) {
    const cudaStream_t stream = slot.stream;
    void *inputRaw = nullptr, *outputRaw = nullptr;
    if (!importPlanePair(vk, state, src, dst, plane, keep, inputRaw, outputRaw, error)) return false;
    const T *input = static_cast<const T *>(inputRaw);
    T *output = static_cast<T *>(outputRaw);
    tcprof::Timer planeTimer(g_prof.plane);

    const int width = vsapi->getFrameWidth(src, plane), height = vsapi->getFrameHeight(src, plane);
    const int stride = static_cast<int>(vsapi->getStride(src, plane) / sizeof(T));
    const int dstStride = static_cast<int>(vsapi->getStride(dst, plane) / sizeof(T));
    const dim3 block(16, 16, 1);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

    // Radii and normalised weights come straight from the CPU setup (uploaded once, see ensureInit).
    const int radiusH = data->radiusH[plane], radiusV = data->radiusV[plane];
    const void *source = input;
    int sourceStride = stride;
    const int temporaryStride = static_cast<int>(state.pitchBytes.load() / sizeof(float));
    if (radiusV > 0) {
        tcanny_blur_v<<<grid, block, 0, stream>>>(static_cast<const T *>(source), static_cast<float *>(slot.bufferV), width, height, sourceStride, temporaryStride,
                                                  radiusV, state.weights + state.offsetV[plane]);
        source = slot.bufferV;
        sourceStride = temporaryStride;
    }
    if (radiusH > 0) {
        // After the vertical pass the source is the float temporary, otherwise the typed input.
        if (radiusV > 0)
            tcanny_blur_h<float><<<grid, block, 0, stream>>>(static_cast<const float *>(source), static_cast<float *>(slot.bufferH), width, height, sourceStride,
                                                             temporaryStride, radiusH, state.weights + state.offsetH[plane]);
        else
            tcanny_blur_h<T><<<grid, block, 0, stream>>>(static_cast<const T *>(source), static_cast<float *>(slot.bufferH), width, height, sourceStride,
                                                         temporaryStride, radiusH, state.weights + state.offsetH[plane]);
        source = slot.bufferH;
        sourceStride = temporaryStride;
    }
    const bool blurred = source != static_cast<const void *>(input);
    // Mode 0 keeps a float state plane: the destination itself for float input, the slot's state buffer for integers.
    float *stateOut = std::is_same<T, float>::value ? reinterpret_cast<float *>(output) : static_cast<float *>(slot.stateBuf);
    const int stateStride = std::is_same<T, float>::value ? dstStride : temporaryStride;
    // Mode 0 stores the gradient magnitudes in whichever blur temporary is not the kernel's input.
    float *magnitude = static_cast<float *>(source == slot.bufferH ? slot.bufferV : slot.bufferH);
    {
        tcprof::Timer t(g_prof.launches);
        tcanny_kernel<T><<<grid, block, 0, stream>>>(
            input, output, stateOut, blurred ? static_cast<const float *>(source) : nullptr, magnitude, width, height, stride, dstStride, stateStride,
            blurred ? sourceStride : 0, temporaryStride, data->mode, data->op, data->scale, data->peak);
        if (!cudaCheck(cudaGetLastError(), "tcanny CUDA launch", error)) return false;
        if (data->mode == 0) {
            tcanny_nms<<<grid, block, 0, stream>>>(stateOut, magnitude, width, height, stateStride, temporaryStride, data->t_h, data->t_l);
            if (!cudaCheck(cudaGetLastError(), "tcanny NMS launch", error)) return false;
        }
    }

    if (data->mode == 0) {
        tcprof::Timer hystTimer(g_prof.hysteresis);
        const int wordsPerRow = (width + 31) / 32;
        tcanny_pack<<<dim3(wordsPerRow, (height + 7) / 8, 1), dim3(32, 8, 1), 0, stream>>>(stateOut, width, height, stateStride, slot.cand, slot.strong, wordsPerRow);
        if (!cudaCheck(cudaGetLastError(), "tcanny pack launch", error)) return false;
        // Iterate promotion passes to the fixed point: a weak chain may be arbitrarily long, so no fixed pass count is
        // sufficient. The first batch is sized from the depths of recent frames (hysteresis_schedule.h), so most frames
        // need one round trip; later batches double. A batch whose flag is below its pass count ended on an idle pass.
        const dim3 tileBlock(32, 1, 1);
        const dim3 tileGrid((wordsPerRow + 31) / 32, (height + kTileRows - 1) / kTileRows, 1);
        int *flag = slot.changed;
        uint32_t batch = forcedBatch() ? forcedBatch() : state.depthHint.firstBatch();
        uint64_t passesDone = 0;
        const uint64_t pixelBound = uint64_t(width) * height;
        for (;;) {
            if (!cudaCheck(cudaMemsetAsync(flag, 0, sizeof(int), stream), "cudaMemsetAsync(changed)", error)) return false;
            for (uint32_t pass = 0; pass < batch; ++pass)
                tcanny_hysteresis_tile<<<tileGrid, tileBlock, 0, stream>>>(slot.cand, slot.strong, height, wordsPerRow, flag, static_cast<int>(pass));
            if (!cudaCheck(cudaGetLastError(), "tcanny hysteresis launch", error)) return false;
            if (!cudaCheck(cudaMemcpyAsync(slot.hostFlag, flag, sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(changed)", error)) return false;
            if (!cudaCheck(cudaStreamSynchronize(stream), "tcanny hysteresis synchronize", error)) return false;
            const uint32_t raised = static_cast<uint32_t>(*static_cast<volatile int *>(slot.hostFlag));
            if (raised < batch) {
                state.depthHint.record(static_cast<uint32_t>(passesDone + raised));
                break;
            }
            passesDone += batch;
            if (passesDone > pixelBound + HysteresisDepthHint::kMaxBatch) { error = "hysteresis did not converge"; return false; }
            batch = forcedBatch() ? forcedBatch() : HysteresisDepthHint::grow(batch);
        }
        tcanny_finalize<T><<<grid, block, 0, stream>>>(output, slot.strong, width, height, dstStride, wordsPerRow, data->peak);
        if (!cudaCheck(cudaGetLastError(), "tcanny finalize launch", error)) return false;
    }
    return true;
}

bool processPlane(const VSVULKANAPI *vk, const VSAPI *vsapi, CudaState &state, CudaSlot &slot, KeepAlive &keep, const TCannyData *data,
                  const VSFrame *src, VSFrame *dst, int plane, std::string &error) {
    const VSVideoFormat &format = data->vi->format;
    if (format.sampleType == stFloat) return processPlaneAs<float>(vk, vsapi, state, slot, keep, data, src, dst, plane, error);
    if (format.bytesPerSample == 1) return processPlaneAs<unsigned char>(vk, vsapi, state, slot, keep, data, src, dst, plane, error);
    return processPlaneAs<unsigned short>(vk, vsapi, state, slot, keep, data, src, dst, plane, error);
}
} // namespace

void tcannyCudaProfDump() noexcept {
#ifdef TCANNY_PROF
    const uint64_t n = g_prof.frames.load();
    if (!tcprof::enabled() || n == 0) return;
    std::fprintf(stderr, "[TCANNY_PROF cuda] frames=%llu (thread-time summed over worker threads; blocking included)\n", (unsigned long long)n);
    tcprof::line("total tcannyProcessCuda", g_prof.total, n);
    tcprof::line("waitGPUFrame", g_prof.wait, n);
    tcprof::line("selectDevice (resolve+current)", g_prof.select, n);
    tcprof::line("  makeCurrent", g_prof.makeCur, n);
    tcprof::line("newGPUVideoFrame", g_prof.newFrame, n);
    tcprof::line("processPlane (after import)", g_prof.plane, n);
    tcprof::line("export+import", g_prof.importPlanes, n);
    tcprof::line("launch blur+tcanny_kernel", g_prof.launches, n);
    tcprof::line("hysteresis loop total", g_prof.hysteresis, n);
    tcprof::line("final stream sync", g_prof.finalSync, n);
    tcprof::line("setGPUPlaneProducer", g_prof.publish, n);
    tcprof::line("slot gate wait (SlotLease)", g_prof.slotWait, n);
    tcprof::line("init (ensureInit + slot setup; first frames dominate)", g_prof.init, n);
#endif
}

bool tcannyCudaCompiled() noexcept { return true; }

void *tcannyCreateCudaState(std::string &error) noexcept {
    try {
        auto *state = new CudaState();
#ifdef TCANNY_PROF
        if (const char *slots = std::getenv("TCANNY_CUDA_SLOTS")) state->maxSlots = static_cast<size_t>(std::max(1, std::atoi(slots)));
#endif
        return state;
    } catch (...) {
        error = "failed to allocate the CUDA state";
        return nullptr;
    }
}

void tcannyDestroyCudaState(void *state) noexcept {
    try {
        delete static_cast<CudaState *>(state);
    } catch (...) {
    }
}

// Contract: returns non-null iff `error` is empty on return.
VSFrame *tcannyProcessCuda(const VSFrame *src, const TCannyData *data, VSCore *core, const VSAPI *vsapi, std::string &error) noexcept {
    try {
        error.clear();
        const VSVideoFormat &inputFormat = data->vi->format;
        if (!((inputFormat.sampleType == stInteger && inputFormat.bitsPerSample >= 8 && inputFormat.bitsPerSample <= 16) ||
              (inputFormat.sampleType == stFloat && inputFormat.bitsPerSample == 32))) {
            error = "CUDA TCanny supports 8-16 bit integer and 32-bit float input";
            return nullptr;
        }
        const VSVULKANAPI *vk = vsapi->getVulkanAPI();
        if (!vk) { error = "VapourSynth Vulkan API is unavailable"; return nullptr; }
        char message[512]{};
        int device = 0;
        g_prof_frames_add();
        tcprof::Timer totalTimer(g_prof.total);
        {
            tcprof::Timer t(g_prof.wait);
            if (vk->waitGPUFrame(src, message, sizeof(message)) != 0) { error = std::string("waitGPUFrame failed: ") + message; return nullptr; }
        }
        auto *state = static_cast<CudaState *>(data->cudaState);
        if (!state) { error = "CUDA state is not initialized"; return nullptr; }
        {
            tcprof::Timer t(g_prof.select);
            device = state->resolvedDevice.load();
            if (device < 0) {
                if (!resolveCoreDevice(vk, core, device, error)) return nullptr;
                state->resolvedDevice.store(device);
            }
            tcprof::Timer mc(g_prof.makeCur);
            if (!makeCurrent(device, error)) return nullptr;
        }
        FrameGuard dst{vsapi, nullptr};
        {
            tcprof::Timer t(g_prof.newFrame);
            dst.frame = vk->newGPUVideoFrame(&data->vi->format, data->vi->width, data->vi->height, src, core);
        }
        if (!dst.frame) { error = "failed to allocate CUDA output frame"; return nullptr; }
        {
            tcprof::Timer t(g_prof.init);
            if (!ensureInit(*state, data, core, device, error)) return nullptr;
        }
        // The slot is taken last, so waiting for one only ever covers CUDA work. Declared after dst: it is released
        // (stream drained) before dst is freed on every exit path.
        // The imports in `keep` are declared before the lease so the stream is drained before they are released.
        KeepAlive keep;
        SlotLease lease(*state, error);
        if (!lease.get() || !error.empty()) return nullptr;
        CudaSlot &slot = *lease.get();
        for (int plane = 0; plane < data->vi->format.numPlanes; ++plane) {
            const bool ok = data->process[plane] ? processPlane(vk, vsapi, *state, slot, keep, data, src, dst.frame, plane, error)
                                                 : copyPlane(vk, vsapi, *state, slot, keep, src, dst.frame, plane, error);
            if (!ok) return nullptr; // dst freed by guard
        }
        {
            tcprof::Timer t(g_prof.finalSync);
            if (!cudaCheck(cudaStreamSynchronize(slot.stream), "tcanny CUDA synchronize", error)) return nullptr;
        }
        // Every plane of dst was written by CUDA and its stream work has completed on the host, so the content is
        // ready immediately: a null producer says exactly that (VSVulkan4.h) and needs no timeline or semaphore.
        {
            tcprof::Timer t(g_prof.publish);
            for (int plane = 0; plane < data->vi->format.numPlanes; ++plane) vk->setGPUPlaneProducer(dst.frame, plane, nullptr, 0);
        }
        error.clear();
        return dst.release();
    } catch (const std::exception &e) {
        error = std::string("CUDA backend exception: ") + e.what();
        return nullptr;
    } catch (...) {
        error = "CUDA backend: unknown exception";
        return nullptr;
    }
}
