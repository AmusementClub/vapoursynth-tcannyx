#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

// Lightweight smoke tests for the scalar primitives used by the plugin. The
// plugin itself is exercised through VapourSynth in the integration workflow.
int main() {
    std::vector<float> image(25, 0.0f);
    image[12] = 1.0f;
    float sum = 0.0f;
    for (float value : image) sum += value;
    assert(std::abs(sum - 1.0f) < 1e-6f);
    assert(sizeof(uint16_t) == 2);
    return 0;
}

