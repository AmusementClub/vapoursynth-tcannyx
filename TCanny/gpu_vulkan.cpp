#include "TCanny.h"
#include "hysteresis_schedule.h"
#include "prof.h"

#include <VSVulkan4.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef TCANNY_PROF
#define PROF_TIMER_OF(state, field) tcprof::Timer profTimer_##field((state)->field)
#else
#define PROF_TIMER_OF(state, field) (void)0
#endif
#define PROF_TIMER(field) PROF_TIMER_OF(s, field)
// Stage labels for nsys (--trace=vulkan,vulkan-annotations); profiling builds only, and only when the core enabled debug utils.
#ifdef TCANNY_PROF
#define PROF_LABEL(name) StageLabel profLabel_##name(*s, times, cmd, #name)
#else
#define PROF_LABEL(name) (void)0
#endif

namespace {

// Main shader. Stages (push constant `stage`):
//   0  gradient / NMS / output straight from the source plane (no blur)
//   1  vertical Gaussian pass   src    -> temp_h   (radius_v == 0: plain int->float copy)
//   2  horizontal Gaussian pass temp_h -> temp_v   (radius_h == 0: plain copy)
//   3  gradient / output from temp_v (modes -1 and 1)
//   4  mode 0: gradient from temp_v:  magnitude -> temp_h, direction (0..3) -> edge
//   5  mode 0: non-maximum suppression: edge (direction) + temp_h (magnitudes) -> edge (state word)
//   6  mode 0: gradient straight from the source plane (no blur), same outputs as stage 4
// Sampling semantics follow the CPU scalar reference: every out-of-range coordinate is
// mirrored without repeating the edge sample (x=-1 -> 1, x=w -> w-2), for the blurred plane,
// for the raw plane when sigma == 0, and for the gradient plane used by NMS.
// In mode 0 the result is written as a per-pixel state word (0 none, 1 weak, 2 strong) into
// `edge`; the pack and hysteresis shaders resolve it and write the destination plane. The gradient is evaluated once
// per pixel (stage 4/6) and stored, and NMS (stage 5) compares the stored magnitudes, like the CPU reference.
constexpr char kShader[] = R"glsl(
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer SourceBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer DestinationBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer FloatBuffer { float data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer WeightBuffer { float data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer EdgeBuffer { uint data[]; };
layout(push_constant, std430) uniform Params {
    SourceBuffer src;
    DestinationBuffer dst;
    FloatBuffer temp_h;
    FloatBuffer temp_v;
    EdgeBuffer edge;
    WeightBuffer weights_h;
    WeightBuffer weights_v;
    uint width;
    uint height;
    uint src_stride;
    uint dst_stride;
    uint bytes_per_sample;
    uint peak;
    int mode;
    int op;
    float scale;
    float high_threshold;
    float low_threshold;
    int radius_h;
    int radius_v;
    uint stage;
} params;
const float kPi = 3.14159265358979323846;
const float kInvPi = 0.318309886183790671538;
int mirror_coord(int value, int limit) {
    if (limit < 2) return 0; // a single sample: nothing to mirror
    if (value < 0) value = -value;
    else if (value >= limit) value = 2 * limit - 2 - value;
    return clamp(value, 0, limit - 1);
}
float load_src(int x, int y) {
    uint byte_offset = uint(y) * params.src_stride + uint(x) * params.bytes_per_sample;
    uint word = params.src.data[byte_offset >> 2];
    uint shift = (byte_offset & 3u) * 8u;
    if (params.bytes_per_sample == 1u) return float((word >> shift) & 255u);
    if (params.bytes_per_sample == 2u) return float((word >> shift) & 65535u);
    return uintBitsToFloat(word);
}
float vertical_pass(int x, int y) {
    int r = params.radius_v;
    if (r == 0) return load_src(x, y);
    precise float sum = 0.0;
    for (int k = -r; k <= r; ++k) {
        precise float s = load_src(x, mirror_coord(y + k, int(params.height)));
        sum += s * params.weights_v.data[k + r];
    }
    return sum;
}
float horizontal_pass(int x, int y) {
    int r = params.radius_h;
    uint row = uint(y) * params.width;
    if (r == 0) return params.temp_h.data[row + uint(x)];
    precise float sum = 0.0;
    for (int k = -r; k <= r; ++k) {
        precise float s = params.temp_h.data[row + uint(mirror_coord(x + k, int(params.width)))];
        sum += s * params.weights_h.data[k + r];
    }
    return sum;
}
float blurred_sample(int x, int y) {
    x = mirror_coord(x, int(params.width));
    y = mirror_coord(y, int(params.height));
    if (params.stage == 0u || params.stage == 6u) return load_src(x, y);
    return params.temp_v.data[uint(y) * params.width + uint(x)];
}
void store_pixel(uint x, uint y, float value) {
    uint byte_offset = y * params.dst_stride + x * params.bytes_per_sample;
    if (params.bytes_per_sample == 4u) {
        params.dst.data[byte_offset >> 2] = floatBitsToUint(value);
        return;
    }
    value = min(floor(value + 0.5), float(params.peak));
    uint shift = (byte_offset & 3u) * 8u;
    uint mask = params.bytes_per_sample == 1u ? 255u : 65535u;
    uint word_index = byte_offset >> 2;
    uint assumed = params.dst.data[word_index];
    for (;;) {
        uint replacement = (assumed & ~(mask << shift)) | (uint(value) << shift);
        uint previous = atomicCompSwap(params.dst.data[word_index], assumed, replacement);
        if (previous == assumed) break;
        assumed = previous;
    }
}
// 8/16-bit destinations with a word-aligned stride are written one 32-bit word (4 or 2 pixels) per invocation, whole,
// instead of per pixel with atomicCompSwap (four or two invocations used to contend for every word). Pixels past the
// width write 0 into the row padding. The host dispatches the store stage over words in that case (same condition).
bool word_store() {
    return params.bytes_per_sample < 4u && (params.dst_stride & 3u) == 0u && (params.stage == 0u || params.stage == 3u);
}
uint quantize(float value) { return uint(min(floor(value + 0.5), float(params.peak))); }
// Correctly rounded float square root. The built-in sqrt is only specified to a few ULP and the
// CPU reference (IEEE sqrtf) decides NMS ties on the last bit, so pick the neighbour of the
// built-in result whose square is closest to x, using the exactly rounded fma residual.
float sqrt_rn(float x) {
    if (!(x > 1.0e-30) || x > 1.0e30) return sqrt(x);
    float s = sqrt(x);
    uint bits = floatBitsToUint(s);
    float lo = uintBitsToFloat(bits - 1u);
    float hi = uintBitsToFloat(bits + 1u);
    float r0 = abs(fma(s, s, -x));
    float r1 = abs(fma(lo, lo, -x));
    float r2 = abs(fma(hi, hi, -x));
    if (r1 < r0 && r1 <= r2) return lo;
    if (r2 < r0) return hi;
    return s;
}
// Scaled gradient magnitude at an in-range pixel; ogx/ogy receive the scaled components.
float gradient_at(int x, int y, out float ogx, out float ogy) {
    precise float gx = 0.0;
    precise float gy = 0.0;
    if (params.op == 6) {
        precise float a1=blurred_sample(x-2,y-2), a2=blurred_sample(x-1,y-2), a3=blurred_sample(x,y-2), a4=blurred_sample(x+1,y-2), a5=blurred_sample(x+2,y-2);
        precise float a6=blurred_sample(x-2,y-1), a7=blurred_sample(x-1,y-1), a8=blurred_sample(x,y-1), a9=blurred_sample(x+1,y-1), a10=blurred_sample(x+2,y-1);
        precise float a11=blurred_sample(x-2,y), a12=blurred_sample(x-1,y), a14=blurred_sample(x+1,y), a15=blurred_sample(x+2,y);
        precise float a16=blurred_sample(x-2,y+1), a17=blurred_sample(x-1,y+1), a18=blurred_sample(x,y+1), a19=blurred_sample(x+1,y+1), a20=blurred_sample(x+2,y+1);
        precise float a21=blurred_sample(x-2,y+2), a22=blurred_sample(x-1,y+2), a23=blurred_sample(x,y+2), a24=blurred_sample(x+1,y+2), a25=blurred_sample(x+2,y+2);
        gx = a5 + 2.0*a10 + 3.0*a15 + 2.0*a20 + a25 + a4 + 2.0*a9 + 3.0*a14 + 2.0*a19 + a24 - a2 - 2.0*a7 - 3.0*a12 - 2.0*a17 - a22 - a1 - 2.0*a6 - 3.0*a11 - 2.0*a16 - a21;
        gy = a1 + 2.0*a2 + 3.0*a3 + 2.0*a4 + a5 + a6 + 2.0*a7 + 3.0*a8 + 2.0*a9 + a10 - a16 - 2.0*a17 - 3.0*a18 - 2.0*a19 - a20 - a21 - 2.0*a22 - 3.0*a23 - 2.0*a24 - a25;
    } else {
        precise float c1 = blurred_sample(x-1,y-1), c2 = blurred_sample(x,y-1), c3 = blurred_sample(x+1,y-1);
        precise float c4 = blurred_sample(x-1,y), c6 = blurred_sample(x+1,y);
        precise float c7 = blurred_sample(x-1,y+1), c8 = blurred_sample(x,y+1), c9 = blurred_sample(x+1,y+1);
        if (params.op == 0) { gx = c6 - c4; gy = c2 - c8; }
        else if (params.op == 2) { gx = c3 + 2.0*c6 + c9 - c1 - 2.0*c4 - c7; gy = c1 + 2.0*c2 + c3 - c7 - 2.0*c8 - c9; }
        else if (params.op == 3) { gx = 3.0*c3 + 10.0*c6 + 3.0*c9 - 3.0*c1 - 10.0*c4 - 3.0*c7; gy = 3.0*c1 + 10.0*c2 + 3.0*c3 - 3.0*c7 - 10.0*c8 - 3.0*c9; }
        else if (params.op == 4) { gx = 17.0*c3 + 61.0*c6 + 17.0*c9 - 17.0*c1 - 61.0*c4 - 17.0*c7; gy = 17.0*c1 + 61.0*c2 + 17.0*c3 - 17.0*c7 - 61.0*c8 - 17.0*c9; }
        else if (params.op == 5) {
            precise float g1 = 5.0*c1 + 5.0*c2 + 5.0*c3 - 3.0*c4 - 3.0*c6 - 3.0*c7 - 3.0*c8 - 3.0*c9;
            precise float g2 = 5.0*c1 + 5.0*c2 - 3.0*c3 + 5.0*c4 - 3.0*c6 - 3.0*c7 - 3.0*c8 - 3.0*c9;
            precise float g3 = 5.0*c1 - 3.0*c2 - 3.0*c3 + 5.0*c4 - 3.0*c6 + 5.0*c7 - 3.0*c8 - 3.0*c9;
            precise float g4 = -3.0*c1 - 3.0*c2 - 3.0*c3 + 5.0*c4 - 3.0*c6 + 5.0*c7 + 5.0*c8 - 3.0*c9;
            precise float g5 = -3.0*c1 - 3.0*c2 - 3.0*c3 - 3.0*c4 - 3.0*c6 + 5.0*c7 + 5.0*c8 + 5.0*c9;
            precise float g6 = -3.0*c1 - 3.0*c2 - 3.0*c3 - 3.0*c4 + 5.0*c6 - 3.0*c7 + 5.0*c8 + 5.0*c9;
            precise float g7 = -3.0*c1 - 3.0*c2 + 5.0*c3 - 3.0*c4 + 5.0*c6 - 3.0*c7 - 3.0*c8 + 5.0*c9;
            precise float g8 = -3.0*c1 + 5.0*c2 + 5.0*c3 - 3.0*c4 + 5.0*c6 - 3.0*c7 - 3.0*c8 - 3.0*c9;
            gx = max(max(max(abs(g1),abs(g2)),max(abs(g3),abs(g4))),max(max(abs(g5),abs(g6)),max(abs(g7),abs(g8))));
        }
        else { gx = (c3 + c6 + c9 - c1 - c4 - c7) / 2.0; gy = (c1 + c2 + c3 - c7 - c8 - c9) / 2.0; }
    }
    if (params.op == 5) {
        precise float kirsch = gx * params.scale;
        ogx = gx; ogy = gy;
        return kirsch;
    }
    gx *= params.scale;
    gy *= params.scale;
    precise float sum_of_squares = gx * gx + gy * gy;
    float magnitude = sqrt_rn(sum_of_squares);
    ogx = gx; ogy = gy;
    return magnitude;
}
// Value of a pixel in modes -1 and 1 before integer rounding.
float output_value(int x, int y) {
    if (params.mode == -1) return blurred_sample(x, y);
    float gx, gy;
    precise float magnitude = gradient_at(x, y, gx, gy);
    if (params.mode == 1 && params.bytes_per_sample == 4u) magnitude = clamp(magnitude, 0.0, 1.0);
    return magnitude;
}
void main() {
    if (word_store()) {
        uint per = 4u / params.bytes_per_sample;
        uint x0 = gl_GlobalInvocationID.x * per, row = gl_GlobalInvocationID.y;
        if (x0 >= params.width || row >= params.height) return;
        uint word = 0u;
        for (uint i = 0u; i < per; ++i)
            if (x0 + i < params.width) word |= quantize(output_value(int(x0 + i), int(row))) << (i * 8u * params.bytes_per_sample);
        params.dst.data[(row * params.dst_stride >> 2) + gl_GlobalInvocationID.x] = word;
        return;
    }
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= params.width || y >= params.height) return;
    if (params.stage == 1u) {
        params.temp_h.data[y * params.width + x] = vertical_pass(int(x), int(y));
        return;
    }
    if (params.stage == 2u) {
        params.temp_v.data[y * params.width + x] = horizontal_pass(int(x), int(y));
        return;
    }
    if (params.stage == 5u) {
        uint index = y * params.width + x;
        int direction = int(params.edge.data[index]);
        ivec2 offset = direction == 0 ? ivec2(1, 0) : direction == 1 ? ivec2(1, -1) : direction == 2 ? ivec2(0, -1) : ivec2(-1, -1);
        int w = int(params.width), h = int(params.height);
        precise float magnitude = params.temp_h.data[index];
        float n1 = params.temp_h.data[uint(mirror_coord(int(y) + offset.y, h)) * params.width + uint(mirror_coord(int(x) + offset.x, w))];
        float n2 = params.temp_h.data[uint(mirror_coord(int(y) - offset.y, h)) * params.width + uint(mirror_coord(int(x) - offset.x, w))];
        uint state = 0u;
        if (magnitude >= max(n1, n2)) {
            if (magnitude >= params.high_threshold) state = 2u;
            else if (magnitude >= params.low_threshold) state = 1u;
        }
        params.edge.data[index] = state;
        return;
    }
    if (params.mode == 0) {
        float gx, gy;
        precise float magnitude = gradient_at(int(x), int(y), gx, gy);
        precise float angle = 0.0;
        if (gx != 0.0 || gy != 0.0) angle = atan(gy, gx);
        if (angle < 0.0) angle += kPi;
        precise float scaled_angle = angle * 4.0 * kInvPi + 0.5;
        int bin = int(scaled_angle);
        int direction = bin >= 4 ? 0 : bin;
        params.temp_h.data[y * params.width + x] = magnitude;
        params.edge.data[y * params.width + x] = uint(direction);
        return;
    }
    store_pixel(x, y, output_value(int(x), int(y)));
}
)glsl";

// Bit packing for hysteresis. One invocation per 32-pixel word of a row: reads the per-pixel state words the main
// shader wrote (0 none, 1 weak, 2 strong) and writes two bitplanes, one bit per pixel (bit i of word wx is the pixel
// x = wx * 32 + i): `cand` (state >= 1) and `strong` (state == 2). Bits at x >= width stay 0. Every output word has
// exactly one writer, so no atomics are needed.
constexpr char kPackShader[] = R"glsl(
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer EdgeBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer BitBuffer { uint data[]; };
layout(push_constant, std430) uniform Params {
    EdgeBuffer edge;
    BitBuffer cand;
    BitBuffer strong;
    uint width;
    uint height;
    uint words_per_row;
} params;
void main() {
    uint wx = gl_GlobalInvocationID.x, y = gl_GlobalInvocationID.y;
    if (wx >= params.words_per_row || y >= params.height) return;
    uint first = wx * 32u;
    uint count = min(32u, params.width - first);
    uint base = y * params.width + first;
    uint c = 0u, s = 0u;
    for (uint b = 0u; b < count; ++b) {
        uint state = params.edge.data[base + b];
        c |= (state != 0u ? 1u : 0u) << b;
        s |= (state == 2u ? 1u : 0u) << b;
    }
    params.cand.data[y * params.words_per_row + wx] = c;
    params.strong.data[y * params.words_per_row + wx] = s;
}
)glsl";

// Final write of hysteresis: the destination plane from the strong bitplane (strong -> peak, everything else -> 0), one
// invocation per pixel. Sub-word destination samples are packed into shared 32-bit words, hence the CAS loop. The push
// block is shared with the tile shader below (the fields it does not read are ignored).
constexpr char kHysteresisShader[] = R"glsl(
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer CandBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer StrongBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer FlagBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer DestinationBuffer { uint data[]; };
layout(push_constant, std430) uniform Params {
    CandBuffer cand;
    StrongBuffer strong;
    FlagBuffer flags;
    DestinationBuffer dst;
    uint width;
    uint height;
    uint words_per_row;
    uint dst_stride;
    uint bytes_per_sample;
    uint peak;
    uint finalize;
    uint pass_index;
} params;
void store_pixel(uint x, uint y, float value) {
    uint byte_offset = y * params.dst_stride + x * params.bytes_per_sample;
    if (params.bytes_per_sample == 4u) { params.dst.data[byte_offset >> 2] = floatBitsToUint(value); return; }
    uint shift = (byte_offset & 3u) * 8u;
    uint mask = params.bytes_per_sample == 1u ? 255u : 65535u;
    uint word_index = byte_offset >> 2;
    uint assumed = params.dst.data[word_index];
    for (;;) {
        uint replacement = (assumed & ~(mask << shift)) | (uint(value) << shift);
        uint previous = atomicCompSwap(params.dst.data[word_index], assumed, replacement);
        if (previous == assumed) break;
        assumed = previous;
    }
}
void main() {
    uint y = gl_GlobalInvocationID.y;
    if (params.bytes_per_sample < 4u && (params.dst_stride & 3u) == 0u) {
        // 8/16-bit, word-aligned stride: one invocation per destination word (4 or 2 pixels), written whole instead of
        // per-pixel atomicCompSwap; pixels past the width write 0 into the row padding. The host dispatches over words.
        uint per = 4u / params.bytes_per_sample;
        uint x0 = gl_GlobalInvocationID.x * per;
        if (x0 >= params.width || y >= params.height) return;
        uint bits = params.strong.data[y * params.words_per_row + (x0 >> 5)] >> (x0 & 31u);
        uint word = 0u;
        for (uint i = 0u; i < per; ++i)
            if (x0 + i < params.width && ((bits >> i) & 1u) != 0u) word |= params.peak << (i * 8u * params.bytes_per_sample);
        params.dst.data[(y * params.dst_stride >> 2) + gl_GlobalInvocationID.x] = word;
        return;
    }
    uint gx = gl_GlobalInvocationID.x;
    if (gx >= params.width || y >= params.height) return;
    uint bit = (params.strong.data[y * params.words_per_row + (gx >> 5)] >> (gx & 31u)) & 1u;
    store_pixel(gx, y, bit != 0u ? float(params.peak) : 0.0);
}
)glsl";

// Hysteresis promotion on bitplanes. The result is the set of candidate pixels that are 8-connected to a strong pixel
// through candidates; it is unique, so any schedule of monotone promotions that stops only when a full pass promotes
// nothing reaches it. Each pass raises flag word 0 to pass_index + 1 if it changed anything (the host zeroes it at the
// start of a batch, and a batch whose flag is below its pass count ended on a pass that promoted nothing: the fixed
// point). Tile sweep, see the file for details.
constexpr char kHysteresisTileShader[] = R"glsl(
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer CandBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer StrongBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer FlagBuffer { uint data[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) coherent buffer DestinationBuffer { uint data[]; };
layout(push_constant, std430) uniform Params {
    CandBuffer cand;
    StrongBuffer strong;
    FlagBuffer flags;
    DestinationBuffer dst;
    uint width;
    uint height;
    uint words_per_row;
    uint dst_stride;
    uint bytes_per_sample;
    uint peak;
    uint finalize;
    uint pass_index;
} params;
// Tile-sweep promotion pass. One invocation owns one 32-pixel word column of kTileRows consecutive rows: it loads the
// tile and a one-row / one-word halo, sweeps down and up (Gauss-Seidel order, with the same per-word update as the
// word pass) until nothing changes or kMaxPairs sweep pairs are done, and writes back the words it changed. Words
// outside the tile (the halo) are read once per pass and may be stale, which only delays a promotion to a later pass.
// The flag gets pass_index + 1 when this invocation changed any word, including when it stopped at the sweep cap.
const uint kTileRows = 8u;
const uint kMaxPairs = 2u;
uint spread(uint n) { return n | (n << 1) | (n >> 1); }
void main() {
    uint wx = gl_GlobalInvocationID.x;
    if (wx >= params.words_per_row) return;
    uint y0 = gl_GlobalInvocationID.y * kTileRows;
    if (y0 >= params.height) return;
    // Index 0 and kTileRows + 1 are the halo rows above and below; rows outside the plane read as 0.
    uint s[kTileRows + 2u];
    uint e[kTileRows + 2u];
    uint c[kTileRows + 2u];
    [[unroll]] for (uint i = 0u; i < kTileRows + 2u; ++i) {
        int row = int(y0) + int(i) - 1;
        uint sv = 0u, ev = 0u, cv = 0u;
        if (row >= 0 && row < int(params.height)) {
            uint base = uint(row) * params.words_per_row;
            sv = params.strong.data[base + wx];
            uint l = wx > 0u ? params.strong.data[base + wx - 1u] : 0u;
            uint r = wx + 1u < params.words_per_row ? params.strong.data[base + wx + 1u] : 0u;
            ev = (l >> 31) | (r << 31);
            if (i >= 1u && i <= kTileRows) cv = params.cand.data[base + wx];
        }
        s[i] = sv;
        e[i] = ev;
        c[i] = cv;
    }
    uint dirty = 0u;
    uint progress = 0u;
    for (uint pair = 0u; pair < kMaxPairs; ++pair) {
        uint before = progress;
        [[unroll]] for (uint k = 0u; k < 2u * kTileRows; ++k) {
            uint i = k < kTileRows ? k + 1u : 2u * kTileRows - k; // 1..kTileRows down, then kTileRows..1 up
            if (c[i] != s[i]) {
                uint around = spread(s[i - 1u]) | e[i - 1u] | spread(s[i]) | e[i] | spread(s[i + 1u]) | e[i + 1u];
                uint t = s[i] | (c[i] & around);
                for (;;) {
                    uint u = t;
                    t |= c[i] & ((t << 1) | (t >> 1));
                    if (t == u) break;
                }
                if (t != s[i]) {
                    s[i] = t;
                    dirty |= 1u << (i - 1u);
                    ++progress;
                }
            }
        }
        if (progress == before) break;
    }
    if (dirty == 0u) return;
    [[unroll]] for (uint i = 1u; i <= kTileRows; ++i)
        if ((dirty & (1u << (i - 1u))) != 0u) params.strong.data[(y0 + i - 1u) * params.words_per_row + wx] = s[i];
    atomicMax(params.flags.data[0], params.pass_index + 1u);
}
)glsl";

struct MainPush {
    VkDeviceAddress src;
    VkDeviceAddress dst;
    VkDeviceAddress tempH;
    VkDeviceAddress tempV;
    VkDeviceAddress edge;
    VkDeviceAddress weightsH;
    VkDeviceAddress weightsV;
    uint32_t width;
    uint32_t height;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t bytesPerSample;
    uint32_t peak;
    int32_t mode;
    int32_t op;
    float scale;
    float highThreshold;
    float lowThreshold;
    int32_t radiusH;
    int32_t radiusV;
    uint32_t stage;
};

struct PackPush {
    VkDeviceAddress edge;
    VkDeviceAddress cand;
    VkDeviceAddress strong;
    uint32_t width;
    uint32_t height;
    uint32_t wordsPerRow;
};

struct HysteresisPush {
    VkDeviceAddress cand;
    VkDeviceAddress strong;
    VkDeviceAddress flag;
    VkDeviceAddress dst;
    uint32_t width;
    uint32_t height;
    uint32_t wordsPerRow;
    uint32_t dstStride;
    uint32_t bytesPerSample;
    uint32_t peak;
    uint32_t finalize;
    uint32_t passIndex;
};

static_assert(sizeof(MainPush) == 112, "MainPush must match the GLSL Params block (std430)");
static_assert(sizeof(PackPush) == 40, "PackPush must match the GLSL Params block (std430)");
static_assert(sizeof(HysteresisPush) == 64, "HysteresisPush must match the GLSL Params block (std430)");
static_assert(sizeof(MainPush) <= 128 && sizeof(PackPush) <= 128 && sizeof(HysteresisPush) <= 128,
              "push constants exceed the guaranteed minimum");

struct VulkanState {
    const VSVULKANAPI *api{};
    const VSVulkanFunctions *vk{};
    VSVulkanCoreHandles handles{};
    VSGPUExecPool *pool{};
    VSGPUShader *shader{};
    VSGPUShader *hysteresisShader{}, *tileShader{};
    VSGPUShader *packShader{};
    VkShaderModule shaderModule{};
    VkShaderModule hysteresisModule{}, tileModule{};
    VkShaderModule packModule{};
    VkPipelineLayout pipelineLayout{};
    VkPipeline pipeline{};
    VkPipelineLayout hysteresisLayout{}, tileLayout{};
    VkPipeline hysteresisPipeline{}, tilePipeline{};
    VkPipelineLayout packLayout{};
    VkPipeline packPipeline{};
    // Gaussian weights of every processed plane, uploaded once at creation: per plane the vertical then the horizontal
    // array, as floats; offsets are in bytes from the start of the buffer.
    VSGPUBuffer *weightsBuffer{};
    VSVulkanBufferInfo weightsInfo{};
    std::array<VkDeviceSize, 3> offsetH{}, offsetV{};
    mutable HysteresisDepthHint depthHint;
#ifdef TCANNY_PROF
    mutable tcprof::Acc total, newFrame, plan, acquire, record, submit, wait, finalize;
    // GPU time per stage from timestamp queries (TCANNY_PROF=1): summed nanoseconds and counts by stage label.
    mutable std::mutex stageMutex;
    mutable std::map<std::string, std::pair<double, uint64_t>> stageNs;
    float timestampPeriod = 1.0f;
    std::atomic<uint64_t> frames{0};
#endif
    mutable tcprof::HystStats hyst;
};

std::string errorText(const char *prefix, const char *detail) {
    return std::string(prefix) + (detail && *detail ? detail : "unknown Vulkan error");
}

// Releases whatever a (possibly partially initialised) state holds. Safe on any prefix of the
// construction sequence in tcannyCreateVulkanState.
void destroyState(VulkanState *state) noexcept {
    if (!state) return;
#ifdef TCANNY_PROF
    if (tcprof::enabled() && state->frames.load() > 0) {
        const uint64_t n = state->frames.load();
        std::fprintf(stderr, "[TCANNY_PROF vulkan] frames=%llu\n", (unsigned long long)n);
        tcprof::line("total tcannyProcessVulkan", state->total, n);
        tcprof::line("newGPUVideoFrame (dst alloc)", state->newFrame, n);
        tcprof::line("plan()", state->plan, n);
        tcprof::line("gpuExecAcquire", state->acquire, n);
        tcprof::line("record() CPU command recording", state->record, n);
        tcprof::line("gpuExecSubmit", state->submit, n);
        tcprof::line("gpuExecWaitValue (blocking)", state->wait, n);
        tcprof::line("finalize record+submit", state->finalize, n);
        state->hyst.print("vulkan");
        std::fprintf(stderr, "  GPU time per stage (timestamp queries, per frame):\n");
        for (const auto &entry : state->stageNs)
            std::fprintf(stderr, "    stage %-12s n/frame=%6.2f  us/frame=%9.2f  us/call=%8.2f\n", entry.first.c_str(), double(entry.second.second) / n,
                         entry.second.first / 1e3 / n, entry.second.second ? entry.second.first / 1e3 / entry.second.second : 0.0);
    }
#endif
    if (state->api && state->pool) state->api->freeGPUExecPool(state->pool);
    if (state->vk && state->handles.device) {
        const VkDevice device = state->handles.device;
        if (state->packPipeline) state->vk->vkDestroyPipeline(device, state->packPipeline, nullptr);
        if (state->packLayout) state->vk->vkDestroyPipelineLayout(device, state->packLayout, nullptr);
        if (state->packModule) state->vk->vkDestroyShaderModule(device, state->packModule, nullptr);
        if (state->tilePipeline) state->vk->vkDestroyPipeline(device, state->tilePipeline, nullptr);
        if (state->tileLayout) state->vk->vkDestroyPipelineLayout(device, state->tileLayout, nullptr);
        if (state->tileModule) state->vk->vkDestroyShaderModule(device, state->tileModule, nullptr);
        if (state->hysteresisPipeline) state->vk->vkDestroyPipeline(device, state->hysteresisPipeline, nullptr);
        if (state->hysteresisLayout) state->vk->vkDestroyPipelineLayout(device, state->hysteresisLayout, nullptr);
        if (state->hysteresisModule) state->vk->vkDestroyShaderModule(device, state->hysteresisModule, nullptr);
        if (state->pipeline) state->vk->vkDestroyPipeline(device, state->pipeline, nullptr);
        if (state->pipelineLayout) state->vk->vkDestroyPipelineLayout(device, state->pipelineLayout, nullptr);
        if (state->shaderModule) state->vk->vkDestroyShaderModule(device, state->shaderModule, nullptr);
    }
    // The pool is gone (drained), so no submission can still read the weights.
    if (state->api && state->weightsBuffer) state->api->destroyGPUBuffer(state->weightsBuffer);
    if (state->api && state->packShader) state->api->freeGPUShader(state->packShader);
    if (state->api && state->hysteresisShader) state->api->freeGPUShader(state->hysteresisShader);
    if (state->api && state->tileShader) state->api->freeGPUShader(state->tileShader);
    if (state->api && state->shader) state->api->freeGPUShader(state->shader);
    delete state;
}

struct StateDeleter {
    void operator()(VulkanState *state) const noexcept { destroyState(state); }
};

bool getAddress(const VulkanState *state, VkBuffer buffer, VkDeviceAddress &address, std::string &error) noexcept {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    address = state->vk->vkGetBufferDeviceAddress(state->handles.device, &info);
    if (!address) {
        error = "Vulkan device does not expose buffer device addresses";
        return false;
    }
    return true;
}

// Builds shader module + pipeline layout + compute pipeline for one compiled shader.
void buildPipeline(VulkanState &state, VSGPUShader *shader, uint32_t pushBytes, const char *what, VkShaderModule &module,
                   VkPipelineLayout &layout, VkPipeline &pipeline) {
    size_t codeBytes = 0;
    const uint32_t *code = state.api->getGPUShaderCode(shader, &codeBytes);
    if (!code || !codeBytes) throw std::runtime_error(std::string("no SPIR-V for ") + what + " shader");
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = codeBytes;
    moduleInfo.pCode = code;
    if (state.vk->vkCreateShaderModule(state.handles.device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error(std::string("vkCreateShaderModule failed for ") + what);
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    if (state.vk->vkCreatePipelineLayout(state.handles.device, &layoutInfo, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error(std::string("vkCreatePipelineLayout failed for ") + what);
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = layout;
    if (state.vk->vkCreateComputePipelines(state.handles.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error(std::string("vkCreateComputePipelines failed for ") + what);
}

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) { return (value + alignment - 1) / alignment * alignment; }

// One processed plane of one frame.
struct PlaneJob {
    int plane{};
    uint32_t width{}, height{}, srcStride{}, dstStride{}, bytesPerSample{}, peak{};
    int radiusH{}, radiusV{};
    bool blur{};
    bool converged{};
    VkDeviceAddress in{}, out{};
    uint32_t wordsPerRow{};                          // bitplane row length in 32-pixel words (mode 0)
    VkDeviceSize offTempH{}, offTempV{}, offEdge{}, offCand{}, offStrong{}; // inside the device scratch buffer
    VkDeviceSize offWeightsH{}, offWeightsV{};      // bytes inside the instance's weights buffer
};

constexpr VkDeviceSize kRegionAlign = 256;
// Rows per tile of the promotion pass (must match kTileRows in tcanny_hysteresis_tile.comp.glsl).
constexpr uint32_t kTileRows = 8;
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

// Everything one getFrame call owns. The destructor releases whatever has not been handed to the
// core, so every early return in run() is leak free:
//  - the open exec context is abandoned (which also releases anything already handed to it),
//  - scratch buffers are destroyed unless a submission may still be using them (handedOver: the
//    context owns them now; leak: a host wait could not establish completion),
//  - the destination frame is freed unless it is being returned.
#ifdef TCANNY_PROF
// Timestamp ranges of one frame's stages; the pool is created on first use and read back once the frame's last submission
// has been waited for (see FrameRun::submitFinal).
struct GpuTimes {
    static constexpr uint32_t kMax = 512;
    VkQueryPool pool{};
    uint32_t count = 0;
    std::vector<std::pair<const char *, uint32_t>> ranges; // label, index of the begin timestamp (end = begin + 1)
};
// A debug-utils label (for nsys annotations) plus, when TCANNY_PROF is set, GPU timestamps around the stage.
struct StageLabel {
    const VulkanState &state;
    GpuTimes &times;
    VkCommandBuffer cmd;
    bool labelled, timed = false;
    uint32_t endIndex = 0;
    StageLabel(const VulkanState &st, GpuTimes &t, VkCommandBuffer c, const char *name)
        : state(st), times(t), cmd(c), labelled(st.vk->vkCmdBeginDebugUtilsLabelEXT && st.vk->vkCmdEndDebugUtilsLabelEXT) {
        if (labelled) {
            VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
            label.pLabelName = name;
            state.vk->vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
        }
        if (!tcprof::enabled()) return;
        if (!times.pool) {
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = GpuTimes::kMax;
            if (state.vk->vkCreateQueryPool(state.handles.device, &info, nullptr, &times.pool) != VK_SUCCESS) { times.pool = VK_NULL_HANDLE; return; }
            state.vk->vkCmdResetQueryPool(cmd, times.pool, 0, GpuTimes::kMax);
        }
        if (times.count + 2 > GpuTimes::kMax) return;
        times.ranges.emplace_back(name, times.count);
        state.vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, times.pool, times.count);
        endIndex = times.count + 1;
        times.count += 2;
        timed = true;
    }
    ~StageLabel() {
        if (timed) state.vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, times.pool, endIndex);
        if (labelled) state.vk->vkCmdEndDebugUtilsLabelEXT(cmd);
    }
};
#endif

struct FrameRun {
    const VulkanState *s;
    const TCannyData *data;
    VSCore *core;
    const VSAPI *vsapi;
    std::string &error;
    VSFrame *dst{};
    VSGPUExecContext *ctx{};
    VSGPUBuffer *scratch{}, *host{};
    VSVulkanBufferInfo scratchInfo{}, hostInfo{};
    bool handedOver{}, leak{};
#ifdef TCANNY_PROF
    GpuTimes times;
#endif
    std::vector<PlaneJob> jobs;
    char message[1024]{};

    FrameRun(const VulkanState *state, const TCannyData *d, VSCore *c, const VSAPI *v, std::string &e)
        : s(state), data(d), core(c), vsapi(v), error(e) {}
    FrameRun(const FrameRun &) = delete;
    FrameRun &operator=(const FrameRun &) = delete;
    ~FrameRun() {
        if (ctx) s->api->gpuExecAbandon(ctx);
        if (!handedOver && !leak) {
            if (scratch) s->api->destroyGPUBuffer(scratch);
            if (host) s->api->destroyGPUBuffer(host);
        }
        if (dst) vsapi->freeFrame(dst);
#ifdef TCANNY_PROF
        if (times.pool && !leak) s->vk->vkDestroyQueryPool(s->handles.device, times.pool, nullptr);
#endif
    }

    // Submits the frame's last command buffer. In profiling runs it waits and reads the stage timestamps back (this
    // changes the timing of that submission, never the result).
    bool submitFinal() {
#ifdef TCANNY_PROF
        if (tcprof::enabled() && times.pool && times.count) {
            uint64_t value = 0;
            if (!submit(&value) || !waitFor(value)) return false;
            std::vector<uint64_t> stamps(times.count);
            if (s->vk->vkGetQueryPoolResults(s->handles.device, times.pool, 0, times.count, stamps.size() * sizeof(uint64_t), stamps.data(), sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) { // every submission has completed: no WAIT_BIT (an unwritten query would hang)
                std::lock_guard<std::mutex> lock(s->stageMutex);
                for (const auto &range : times.ranges) {
                    auto &entry = s->stageNs[range.first];
                    entry.first += double(stamps[range.second + 1] - stamps[range.second]) * s->timestampPeriod;
                    ++entry.second;
                }
            }
            return true;
        }
#endif
        return submit(nullptr);
    }

    bool fail(const std::string &text) { error = text; return false; }

    void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                 VkAccessFlags2 dstAccess) const {
        VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        memory.srcStageMask = srcStage;
        memory.srcAccessMask = srcAccess;
        memory.dstStageMask = dstStage;
        memory.dstAccessMask = dstAccess;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        s->vk->vkCmdPipelineBarrier2(cmd, &dependency);
    }
    void computeBarrier(VkCommandBuffer cmd) const {
        barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    void copyBuffer(VkCommandBuffer cmd, VkBuffer from, VkBuffer to, VkDeviceSize fromOffset, VkDeviceSize toOffset, VkDeviceSize size) const {
        VkBufferCopy2 region{VK_STRUCTURE_TYPE_BUFFER_COPY_2};
        region.srcOffset = fromOffset;
        region.dstOffset = toOffset;
        region.size = size;
        VkCopyBufferInfo2 copy{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
        copy.srcBuffer = from;
        copy.dstBuffer = to;
        copy.regionCount = 1;
        copy.pRegions = &region;
        s->vk->vkCmdCopyBuffer2(cmd, &copy);
    }
    void dispatch(VkCommandBuffer cmd, const PlaneJob &job) const {
        s->vk->vkCmdDispatch(cmd, (job.width + 15) / 16, (job.height + 15) / 16, 1);
    }
    // The bitplane shaders use 8x8 groups: over words for pack and passes, over pixels for finalize.
    void dispatchWords(VkCommandBuffer cmd, const PlaneJob &job) const {
        s->vk->vkCmdDispatch(cmd, (job.wordsPerRow + 7) / 8, (job.height + 7) / 8, 1);
    }
    // One invocation per word column of kTileRows rows (the tile shader's local size is 32x1: one warp per group).
    void dispatchTiles(VkCommandBuffer cmd, const PlaneJob &job) const {
        s->vk->vkCmdDispatch(cmd, (job.wordsPerRow + 31) / 32, (job.height + kTileRows - 1) / kTileRows, 1);
    }
    void dispatchPixels8(VkCommandBuffer cmd, const PlaneJob &job) const {
        s->vk->vkCmdDispatch(cmd, (job.width + 7) / 8, (job.height + 7) / 8, 1);
    }
    // 8/16-bit destinations with a word-aligned stride are written one 32-bit word per invocation (the shaders test the
    // same condition): the store stage and finalize then dispatch over destination words instead of pixels.
    static bool wordStore(const PlaneJob &job) { return job.bytesPerSample < 4 && job.dstStride % 4 == 0; }
    static uint32_t storeColumns(const PlaneJob &job) {
        return wordStore(job) ? (job.width * job.bytesPerSample + 3) / 4 : job.width;
    }
    void pushPack(VkCommandBuffer cmd, const PackPush &push) const {
        VkPushConstantsInfo info{VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO};
        info.layout = s->packLayout;
        info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        info.size = sizeof(push);
        info.pValues = &push;
        s->vk->vkCmdPushConstants2(cmd, &info);
    }
    void pushMain(VkCommandBuffer cmd, const MainPush &push) const {
        VkPushConstantsInfo info{VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO};
        info.layout = s->pipelineLayout;
        info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        info.size = sizeof(push);
        info.pValues = &push;
        s->vk->vkCmdPushConstants2(cmd, &info);
    }
    void pushHysteresis(VkCommandBuffer cmd, const HysteresisPush &push, bool tile = false) const {
        VkPushConstantsInfo info{VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO};
        info.layout = tile ? s->tileLayout : s->hysteresisLayout;
        info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        info.size = sizeof(push);
        info.pValues = &push;
        s->vk->vkCmdPushConstants2(cmd, &info);
    }

    bool acquire() {
        PROF_TIMER(acquire);
        ctx = s->api->gpuExecAcquire(s->pool, message, sizeof(message));
        if (!ctx) return fail(errorText("Vulkan execution acquire failed: ", message));
        return true;
    }
    // Consumes the context whether or not submission succeeds.
    bool submit(uint64_t *value) {
        PROF_TIMER(submit);
        VSGPUExecContext *c = ctx;
        ctx = nullptr;
        if (s->api->gpuExecSubmit(c, value, message, sizeof(message)) != 0) return fail(errorText("Vulkan execution submit failed: ", message));
        return true;
    }
    bool waitFor(uint64_t value) {
        PROF_TIMER(wait);
        const int result = s->api->gpuExecWaitValue(s->pool, value, message, sizeof(message));
        if (result == gdDrained) return true;
        if (!vsGPUDrainSafeToDestroy(result)) leak = true; // the GPU may still use the scratch buffers: never free them
        return fail(errorText("Vulkan wait failed: ", message));
    }

    // Per-plane geometry, scratch layout, scratch/host buffers and weights upload staging.
    bool plan(const VSFrame *src) {
        const int numPlanes = data->vi->format.numPlanes;
        const uint32_t bytesPerSample = static_cast<uint32_t>(data->vi->format.bytesPerSample);
        const uint32_t peak = data->vi->format.sampleType == stInteger ? (1u << data->vi->format.bitsPerSample) - 1u : 1u;
        const bool hysteresis = data->mode == 0;
        VkDeviceSize flagBytes = alignUp(sizeof(uint32_t) * static_cast<VkDeviceSize>(numPlanes), kRegionAlign);
        VkDeviceSize cursor = flagBytes;
        for (int plane = 0; plane < numPlanes; ++plane) {
            if (!data->process[plane]) continue;
            PlaneJob job;
            job.plane = plane;
            VSVulkanPlaneInfo inInfo{}, outInfo{};
            if (s->api->getGPUPlane(src, plane, &inInfo) != 0 || s->api->getGPUPlane(dst, plane, &outInfo) != 0)
                return fail("Vulkan frame plane lookup failed");
            if (!getAddress(s, inInfo.buffer, job.in, error) || !getAddress(s, outInfo.buffer, job.out, error)) return false;
            job.width = static_cast<uint32_t>(vsapi->getFrameWidth(src, plane));
            job.height = static_cast<uint32_t>(vsapi->getFrameHeight(src, plane));
            job.srcStride = static_cast<uint32_t>(vsapi->getStride(src, plane));
            job.dstStride = static_cast<uint32_t>(vsapi->getStride(dst, plane));
            job.bytesPerSample = bytesPerSample;
            job.peak = peak;
            if (!job.width || !job.height || !job.srcStride || !job.dstStride) return fail("invalid Vulkan frame geometry");
            // The shaders read and write whole 32-bit words, so rows must keep words inside the buffer.
            if ((job.srcStride & 3u) || (job.dstStride & 3u)) return fail("unsupported Vulkan plane stride (not a multiple of 4 bytes)");
            if (job.srcStride < job.width * bytesPerSample || job.dstStride < job.width * bytesPerSample)
                return fail("Vulkan plane stride is smaller than the row");
            if (inInfo.bufferSize < VkDeviceSize(job.srcStride) * job.height || outInfo.bufferSize < VkDeviceSize(job.dstStride) * job.height)
                return fail("Vulkan plane buffer is smaller than stride * height");
            job.radiusH = data->radiusH[plane];
            job.radiusV = data->radiusV[plane];
            if (job.radiusH < 0 || job.radiusV < 0) return fail("invalid blur radius");
            if (job.radiusH && (static_cast<size_t>(job.radiusH) * 2 + 1 != data->weightsH[plane].size() || static_cast<uint32_t>(job.radiusH) >= job.width))
                return fail("horizontal blur radius does not fit the plane");
            if (job.radiusV && (static_cast<size_t>(job.radiusV) * 2 + 1 != data->weightsV[plane].size() || static_cast<uint32_t>(job.radiusV) >= job.height))
                return fail("vertical blur radius does not fit the plane");
            job.blur = job.radiusH > 0 || job.radiusV > 0;
            const VkDeviceSize pixels = VkDeviceSize(job.width) * job.height;
            if (job.blur) {
                job.offTempH = cursor; cursor = alignUp(cursor + pixels * sizeof(float), kRegionAlign);
                job.offTempV = cursor; cursor = alignUp(cursor + pixels * sizeof(float), kRegionAlign);
            } else if (hysteresis) {
                // Mode 0 without blur still needs a plane for the stored gradient magnitudes.
                job.offTempH = cursor; cursor = alignUp(cursor + pixels * sizeof(float), kRegionAlign);
            }
            if (hysteresis) {
                job.offEdge = cursor; cursor = alignUp(cursor + pixels * sizeof(uint32_t), kRegionAlign);
                job.wordsPerRow = (job.width + 31u) / 32u;
                const VkDeviceSize bitBytes = VkDeviceSize(job.wordsPerRow) * job.height * sizeof(uint32_t);
                job.offCand = cursor; cursor = alignUp(cursor + bitBytes, kRegionAlign);
                job.offStrong = cursor; cursor = alignUp(cursor + bitBytes, kRegionAlign);
            }
            job.offWeightsH = s->offsetH[plane];
            job.offWeightsV = s->offsetV[plane];
            jobs.push_back(job);
        }
        if (jobs.empty()) return fail("no plane to process");
        scratch = s->api->createGPUBuffer(core, cursor,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &scratchInfo, message, sizeof(message));
        if (!scratch || !scratchInfo.buffer || !scratchInfo.address)
            return fail(errorText("Vulkan scratch buffer allocation failed: ", message));
        if (hysteresis) {
            // Only the convergence flags come back to the host (modes -1 and 1 need no host buffer at all).
            host = s->api->createGPUBuffer(core, flagBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &hostInfo, message, sizeof(message));
            if (!host || !hostInfo.buffer || !hostInfo.mapped) return fail(errorText("Vulkan host readback buffer allocation failed: ", message));
            std::memset(hostInfo.mapped, 0, static_cast<size_t>(flagBytes));
        }
        return true;
    }

    VkDeviceAddress scratchAddress(VkDeviceSize offset) const { return scratchInfo.address + offset; }
    // The flag word of job index i lives at scratch offset 4*i (and the same offset in the host readback buffer).
    VkDeviceAddress flagAddress(size_t i) const { return scratchInfo.address + sizeof(uint32_t) * i; }

    // Records the blur/gradient work (first == true) and, for mode 0, one batch of hysteresis passes
    // followed by a copy of the per-plane "promoted something in the last pass" flags to the host.
    void record(VkCommandBuffer cmd, bool first, uint32_t batch) {
        const VkPipelineStageFlags2 compute = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        const bool hysteresis = data->mode == 0;
        // Make everything an earlier (host-waited) submission wrote visible to this one.
        computeBarrier(cmd);
        if (first) {
            s->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
            for (const PlaneJob &job : jobs) {
                const VkDeviceAddress weightsBase = s->weightsInfo.address;
                MainPush push{job.in, job.out,
                              job.offTempH ? scratchAddress(job.offTempH) : 0, job.blur ? scratchAddress(job.offTempV) : 0,
                              hysteresis ? scratchAddress(job.offEdge) : 0,
                              job.radiusH ? weightsBase + job.offWeightsH : 0, job.radiusV ? weightsBase + job.offWeightsV : 0,
                              job.width, job.height, job.srcStride, job.dstStride, job.bytesPerSample, job.peak,
                              data->mode, data->op, data->scale, data->t_h, data->t_l, job.radiusH, job.radiusV, 0u};
                if (job.blur) {
                    {
                        PROF_LABEL(blur_v);
                        push.stage = 1;
                        pushMain(cmd, push);
                        dispatch(cmd, job);
                        computeBarrier(cmd);
                    }
                    {
                        PROF_LABEL(blur_h);
                        push.stage = 2;
                        pushMain(cmd, push);
                        dispatch(cmd, job);
                        computeBarrier(cmd);
                    }
                }
                if (hysteresis) {
                    {
                        PROF_LABEL(gradient);
                        push.stage = job.blur ? 4u : 6u; // gradient once per pixel
                        pushMain(cmd, push);
                        dispatch(cmd, job);
                        computeBarrier(cmd);
                    }
                    {
                        PROF_LABEL(nms);
                        push.stage = 5u; // non-maximum suppression on the stored magnitudes
                        pushMain(cmd, push);
                        dispatch(cmd, job);
                        computeBarrier(cmd);
                    }
                } else {
                    PROF_LABEL(store);
                    push.stage = job.blur ? 3u : 0u;
                    pushMain(cmd, push);
                    s->vk->vkCmdDispatch(cmd, (storeColumns(job) + 15) / 16, (job.height + 15) / 16, 1);
                    computeBarrier(cmd);
                }
            }
        }
        if (!hysteresis) return;

        if (first) {
            // Turn the per-pixel state words into the candidate and strong bitplanes (every word is written).
            PROF_LABEL(pack);
            s->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->packPipeline);
            for (const PlaneJob &job : jobs) {
                PackPush push{scratchAddress(job.offEdge), scratchAddress(job.offCand), scratchAddress(job.offStrong), job.width, job.height,
                              job.wordsPerRow};
                pushPack(cmd, push);
                dispatchWords(cmd, job);
            }
            computeBarrier(cmd);
        }
        // Zero the per-plane flags for this batch (they were read by the host after the previous submission).
        barrier(cmd, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        s->vk->vkCmdFillBuffer(cmd, scratchInfo.buffer, 0, sizeof(uint32_t) * jobs.size(), 0);
        barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, compute,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        PROF_LABEL(hysteresis);
        s->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->tilePipeline);
        for (uint32_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < jobs.size(); ++j) {
                const PlaneJob &job = jobs[j];
                if (job.converged) continue;
                HysteresisPush push{scratchAddress(job.offCand), scratchAddress(job.offStrong), flagAddress(j), job.out, job.width, job.height,
                                    job.wordsPerRow, job.dstStride, job.bytesPerSample, job.peak, 0u, i};
                pushHysteresis(cmd, push, true);
                dispatchTiles(cmd, job);
            }
            computeBarrier(cmd);
        }
        barrier(cmd, compute, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        copyBuffer(cmd, scratchInfo.buffer, hostInfo.buffer, 0, 0, sizeof(uint32_t) * jobs.size());
        barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    }

    void handOverBuffers() {
        s->api->gpuExecUsesBuffer(ctx, scratch);
        if (host) s->api->gpuExecUsesBuffer(ctx, host);
        handedOver = true;
    }

    // Returns the finished frame (ownership passes to the caller) or nullptr with `error` set.
    VSFrame *run(const VSFrame *src) {
        const bool allPlanes = std::all_of(data->process.begin(), data->process.begin() + data->vi->format.numPlanes, [](bool v) { return v; });
        const int planeMap[] = {0, 1, 2};
        const VSFrame *sources[] = {data->process[0] ? nullptr : src, data->process[1] ? nullptr : src, data->process[2] ? nullptr : src};
        {
            PROF_TIMER(newFrame);
            if (allPlanes) dst = s->api->newGPUVideoFrame(&data->vi->format, data->vi->width, data->vi->height, src, core);
            else dst = vsapi->newVideoFrame2(&data->vi->format, data->vi->width, data->vi->height, sources, planeMap, src, core);
        }
        if (!dst) { error = "failed to allocate Vulkan output frame"; return nullptr; }
        {
            PROF_TIMER(plan);
            if (!plan(src)) return nullptr;
        }

        const bool hysteresis = data->mode == 0;
        uint32_t batch = forcedBatch() ? forcedBatch() : s->depthHint.firstBatch();
        uint64_t pixelBound = 0;
        for (const PlaneJob &job : jobs) pixelBound = std::max<uint64_t>(pixelBound, uint64_t(job.width) * job.height);
        uint64_t passesDone = 0, frameDepth = 0;
        uint32_t profBatches = 0, profPasses = 0;
        bool first = true;
        while (true) {
            if (!acquire()) return nullptr;
            VkCommandBuffer cmd = s->api->gpuExecCommandBuffer(ctx);
            if (first) s->api->gpuExecReadsFrame(ctx, src);
            if (!hysteresis) {
                for (const PlaneJob &job : jobs) s->api->gpuExecWritesPlane(ctx, dst, job.plane);
                handOverBuffers();
            }
            {
                PROF_TIMER(record);
                record(cmd, first, batch);
            }
            ++profBatches;
            profPasses += hysteresis ? batch : 0;
            if (!hysteresis) {
                if (!submitFinal()) return nullptr;
                VSFrame *result = dst;
                dst = nullptr;
                return result;
            }
            // Mode 0: this submission only touches scratch (and reads src); wait for it, read the
            // flags, and go around again until a batch's last pass promotes nothing.
            uint64_t signaled = 0;
            if (!submit(&signaled) || !waitFor(signaled)) return nullptr;
            const auto *flags = static_cast<const volatile uint32_t *>(hostInfo.mapped);
            bool allConverged = true;
            for (size_t j = 0; j < jobs.size(); ++j) {
                const uint32_t raised = flags[j]; // 0 when the plane was already converged (not dispatched) or promoted nothing
                if (!jobs[j].converged && raised < batch) {
                    jobs[j].converged = true;
                    frameDepth = std::max<uint64_t>(frameDepth, passesDone + raised); // exact: passes up to the last promotion
                }
                allConverged = allConverged && jobs[j].converged;
            }
            if (allConverged) {
                s->hyst.add(profBatches, profPasses);
                s->depthHint.record(static_cast<uint32_t>(frameDepth));
                break;
            }
            passesDone += batch;
            // Every batch that has not converged promotes at least one pixel, so this cannot trigger
            // on a correct run; it only keeps a misbehaving device from looping forever.
            if (passesDone > pixelBound + HysteresisDepthHint::kMaxBatch) return fail("hysteresis did not converge"), nullptr;
            first = false;
            batch = forcedBatch() ? forcedBatch() : HysteresisDepthHint::grow(batch);
        }

        // Finalize: write the destination planes from the resolved bitplanes.
        PROF_TIMER(finalize);
        if (!acquire()) return nullptr;
        VkCommandBuffer cmd = s->api->gpuExecCommandBuffer(ctx);
        for (const PlaneJob &job : jobs) s->api->gpuExecWritesPlane(ctx, dst, job.plane);
        handOverBuffers();
        computeBarrier(cmd);
        { // scoped: the label must end before the command buffer is submitted
            PROF_LABEL(finalize);
            s->vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->hysteresisPipeline);
            for (size_t j = 0; j < jobs.size(); ++j) {
                const PlaneJob &job = jobs[j];
                HysteresisPush push{scratchAddress(job.offCand), scratchAddress(job.offStrong), flagAddress(j), job.out, job.width, job.height,
                                    job.wordsPerRow, job.dstStride, job.bytesPerSample, job.peak, 1u, 0u};
                pushHysteresis(cmd, push);
                s->vk->vkCmdDispatch(cmd, (storeColumns(job) + 7) / 8, (job.height + 7) / 8, 1);
            }
        }
        if (!submitFinal()) return nullptr;
        VSFrame *result = dst;
        dst = nullptr;
        return result;
    }
};

} // namespace

namespace {
// Uploads the Gaussian weights of every processed plane into one device-local buffer, once, and waits for the copy
// (one-shot setup work, which is what gpuExecWaitValue is for). Frames then only carry the buffer's address.
void uploadWeights(VulkanState &state, VSCore *core, const TCannyData *data) {
    std::vector<float> host;
    for (int plane = 0; plane < data->vi->format.numPlanes; ++plane) {
        if (!data->process[plane]) continue;
        const int radiusH = data->radiusH[plane], radiusV = data->radiusV[plane];
        if (radiusV > 0 && data->weightsV[plane].size() != size_t(2 * radiusV + 1)) throw std::runtime_error("inconsistent vertical Gaussian weights");
        if (radiusH > 0 && data->weightsH[plane].size() != size_t(2 * radiusH + 1)) throw std::runtime_error("inconsistent horizontal Gaussian weights");
        state.offsetV[plane] = host.size() * sizeof(float);
        if (radiusV > 0) host.insert(host.end(), data->weightsV[plane].begin(), data->weightsV[plane].end());
        state.offsetH[plane] = host.size() * sizeof(float);
        if (radiusH > 0) host.insert(host.end(), data->weightsH[plane].begin(), data->weightsH[plane].end());
    }
    if (host.empty()) return;
    const VkDeviceSize bytes = host.size() * sizeof(float);
    char message[1024]{};
    state.weightsBuffer = state.api->createGPUBuffer(core, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &state.weightsInfo, message, sizeof(message));
    if (!state.weightsBuffer || !state.weightsInfo.buffer || !state.weightsInfo.address)
        throw std::runtime_error(errorText("Vulkan weights buffer allocation failed: ", message));
    VSVulkanBufferInfo stagingInfo{};
    VSGPUBuffer *staging = state.api->createGPUBuffer(core, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &stagingInfo, message, sizeof(message));
    if (!staging || !stagingInfo.buffer || !stagingInfo.mapped) {
        if (staging) state.api->destroyGPUBuffer(staging);
        throw std::runtime_error(errorText("Vulkan weights staging buffer allocation failed: ", message));
    }
    std::memcpy(stagingInfo.mapped, host.data(), static_cast<size_t>(bytes));
    VSGPUExecContext *ctx = state.api->gpuExecAcquire(state.pool, message, sizeof(message));
    if (!ctx) {
        state.api->destroyGPUBuffer(staging);
        throw std::runtime_error(errorText("Vulkan execution acquire failed: ", message));
    }
    VkCommandBuffer cmd = state.api->gpuExecCommandBuffer(ctx);
    VkBufferCopy2 region{VK_STRUCTURE_TYPE_BUFFER_COPY_2};
    region.size = bytes;
    VkCopyBufferInfo2 copy{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    copy.srcBuffer = stagingInfo.buffer;
    copy.dstBuffer = state.weightsInfo.buffer;
    copy.regionCount = 1;
    copy.pRegions = &region;
    state.vk->vkCmdCopyBuffer2(cmd, &copy);
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    state.vk->vkCmdPipelineBarrier2(cmd, &dependency);
    // The staging buffer belongs to the submission from here on (released once it completed).
    state.api->gpuExecUsesBuffer(ctx, staging);
    uint64_t value = 0;
    if (state.api->gpuExecSubmit(ctx, &value, message, sizeof(message)) != 0)
        throw std::runtime_error(errorText("Vulkan weights upload submit failed: ", message));
    if (state.api->gpuExecWaitValue(state.pool, value, message, sizeof(message)) != gdDrained)
        throw std::runtime_error(errorText("Vulkan weights upload wait failed: ", message));
}
} // namespace

void *tcannyCreateVulkanState(VSCore *core, const VSAPI *vsapi, const TCannyData *data, std::string &error) noexcept {
    std::unique_ptr<VulkanState, StateDeleter> state;
    try {
        state.reset(new VulkanState);
        state->api = vsapi->getVulkanAPI();
        if (!state->api) throw std::runtime_error("this VapourSynth core has no Vulkan API");
        char message[1024]{};
        if (state->api->getVulkanHandles(core, &state->handles, message, sizeof(message)) != 0)
            throw std::runtime_error(errorText("Vulkan handles unavailable: ", message));
        state->vk = state->api->getVulkanFunctions(core, message, sizeof(message));
        if (!state->vk) throw std::runtime_error(errorText("Vulkan function table unavailable: ", message));
#ifdef TCANNY_PROF
        if (const auto getProps = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                state->handles.getInstanceProcAddr(state->handles.instance, "vkGetPhysicalDeviceProperties"))) {
            VkPhysicalDeviceProperties props{};
            getProps(state->handles.physicalDevice, &props);
            state->timestampPeriod = props.limits.timestampPeriod;
        }
#endif
        state->shader = state->api->compileGPUShader(core, slGLSL, kShader, message, sizeof(message));
        if (!state->shader) throw std::runtime_error(errorText("TCanny Vulkan shader compilation failed: ", message));
        buildPipeline(*state, state->shader, sizeof(MainPush), "main", state->shaderModule, state->pipelineLayout, state->pipeline);
        state->hysteresisShader = state->api->compileGPUShader(core, slGLSL, kHysteresisShader, message, sizeof(message));
        if (!state->hysteresisShader) throw std::runtime_error(errorText("TCanny hysteresis shader compilation failed: ", message));
        buildPipeline(*state, state->hysteresisShader, sizeof(HysteresisPush), "hysteresis", state->hysteresisModule, state->hysteresisLayout,
                      state->hysteresisPipeline);
        state->tileShader = state->api->compileGPUShader(core, slGLSL, kHysteresisTileShader, message, sizeof(message));
        if (!state->tileShader) throw std::runtime_error(errorText("TCanny hysteresis tile shader compilation failed: ", message));
        buildPipeline(*state, state->tileShader, sizeof(HysteresisPush), "hysteresis tile", state->tileModule, state->tileLayout, state->tilePipeline);
        state->packShader = state->api->compileGPUShader(core, slGLSL, kPackShader, message, sizeof(message));
        if (!state->packShader) throw std::runtime_error(errorText("TCanny pack shader compilation failed: ", message));
        buildPipeline(*state, state->packShader, sizeof(PackPush), "pack", state->packModule, state->packLayout, state->packPipeline);
        state->pool = state->api->createGPUExecPool(core, vqCompute, message, sizeof(message));
        if (!state->pool) throw std::runtime_error(errorText("Vulkan execution pool creation failed: ", message));
        uploadWeights(*state, core, data);
        return state.release();
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr; // `state` releases whatever was built so far
    } catch (...) {
        error = "unknown error while creating the Vulkan state";
        return nullptr;
    }
}

void tcannyDestroyVulkanState(void *opaque) noexcept {
    destroyState(static_cast<VulkanState *>(opaque));
}

VSFrame *tcannyProcessVulkan(const VSFrame *src, const TCannyData *data, VSCore *core, const VSAPI *vsapi,
                             std::string &error) noexcept {
    auto *state = static_cast<VulkanState *>(data->vulkanState);
    if (!state || !state->pool) { error = "Vulkan execution state is not initialized"; return nullptr; }
    try {
        PROF_TIMER_OF(state, total);
#ifdef TCANNY_PROF
        state->frames.fetch_add(1);
#endif
        FrameRun run(state, data, core, vsapi, error);
        return run.run(src);
    } catch (const std::exception &e) {
        error = e.what();
    } catch (...) {
        error = "unknown error in the Vulkan backend";
    }
    return nullptr;
}
