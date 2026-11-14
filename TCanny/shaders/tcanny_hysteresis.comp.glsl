
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
