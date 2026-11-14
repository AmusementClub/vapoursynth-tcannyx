
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
