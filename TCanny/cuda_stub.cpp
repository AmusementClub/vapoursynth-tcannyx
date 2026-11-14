#include "TCanny.h"

#if !defined(TCANNY_WITH_CUDA)
void *tcannyCreateCudaState(std::string &error) noexcept {
    error = "CUDA support was not compiled into this build";
    return nullptr;
}
void tcannyDestroyCudaState(void *) noexcept {}
void tcannyCudaProfDump() noexcept {}
bool tcannyCudaCompiled() noexcept { return false; }
VSFrame *tcannyProcessCuda(const VSFrame *, const TCannyData *, VSCore *, const VSAPI *, std::string &error) noexcept {
    error = "CUDA backend was not compiled in this build";
    return nullptr;
}
#endif

