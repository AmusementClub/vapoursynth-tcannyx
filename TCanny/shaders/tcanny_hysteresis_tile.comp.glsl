
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
