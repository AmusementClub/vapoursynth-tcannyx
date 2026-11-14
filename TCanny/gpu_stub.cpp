#include "TCanny.h"

void *tcannyCreateVulkanState(VSCore *, const VSAPI *, const TCannyData *, std::string &error) noexcept {
    error = "Vulkan backend was not compiled in this build";
    return nullptr;
}

void tcannyDestroyVulkanState(void *) noexcept {}

VSFrame *tcannyProcessVulkan(const VSFrame *, const TCannyData *, VSCore *, const VSAPI *, std::string &error) noexcept {
    error = "Vulkan backend was not compiled in this build";
    return nullptr;
}

