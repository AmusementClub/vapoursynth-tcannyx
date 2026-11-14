
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
