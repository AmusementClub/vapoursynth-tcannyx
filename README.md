# TCanny (tcannyx)

Canny edge detection for [VapourSynth](https://www.vapoursynth.com/), with a CPU backend and two GPU backends (Vulkan
and CUDA).

It is a drop-in replacement for [TCanny r14](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny): same
arguments, same output. It loads next to the original, so you can compare the two or switch back at any time.

```python
import vapoursynth as vs
core = vs.core

edges = core.tcannyx.TCanny(clip, sigma=1.5, mode=0)
```

Highlights:
- **Same results as TCanny r14.** The CPU output is bit-identical to the original, and both GPU backends match the
  CPU reference exactly on the test host.
- **Faster on the CPU.** Edge maps (`mode=0`) are 2.2-2.8x faster than the original. Gradients and blurs are as fast or
  slightly faster.
- **ARM support.** NEON on Apple Silicon and Linux arm64 is 3.2-3.9x faster than plain C.
- **GPU processing** for clips that already live on the GPU (VapourSynth's GPU frames), with no hidden copies.

## Requirements

- VapourSynth R80 or newer. The GPU backends need API 4.3 (R81 or newer).
- For `backend="vulkan"`: a Vulkan-capable GPU.
- For `backend="cuda"`: Linux, and an NVIDIA GPU with a recent driver.

## Installation

Download the package for your platform from the releases page and copy the plugin file (`libtcannyx.so`,
`tcannyx.dll` or `libtcannyx.dylib`) into your VapourSynth plugin folder, or load it explicitly:

```python
core.std.LoadPlugin("/path/to/libtcannyx.so")
```

| Package | Backends |
|---|---|
| `linux-x64-cpu`, `windows-x64-cpu` | CPU |
| `linux-x64-vulkan`, `windows-x64-vulkan` | CPU, Vulkan |
| `linux-x64-cuda` | CPU, Vulkan, CUDA |
| `linux-arm64-cpu`, `macos-arm64-cpu` | CPU (NEON) |

The plugin lives in the `tcannyx` namespace, and its file name does not clash with the original TCanny's `libtcanny`,
so the two can be installed side by side.

## Usage

### Edge map (default)

```python
edges = core.tcannyx.TCanny(clip)                          # Canny edges, sigma 1.5, Prewitt operator
edges = core.tcannyx.TCanny(clip, sigma=2.0, t_h=12, t_l=3)
edges = core.tcannyx.TCanny(clip, planes=[0])              # luma only; the other planes are copied
```

### Gradient magnitude or blur only

```python
grad = core.tcannyx.TCanny(clip, mode=1, op=2)             # Sobel gradient magnitude
blur = core.tcannyx.TCanny(clip, mode=-1, sigma=3.0)       # Gaussian blur only
```

### On the GPU

The GPU backends work on GPU-resident clips. The plugin never copies frames between CPU and GPU behind your back:
place `std.GPUUpload()` and `std.GPUDownload()` where you want the copies to happen.

```python
gpu = clip.std.GPUUpload()
edges = core.tcannyx.TCanny(gpu, mode=0)                   # backend="auto" picks Vulkan for GPU clips
edges = core.tcannyx.TCanny(gpu, mode=0, backend="cuda")   # or ask for CUDA explicitly
result = edges.std.GPUDownload()
```

The GPU pays off most when several GPU filters run back to back, so the clip stays on the GPU. For a single filter
between an upload and a download, the copies usually take longer than the filter itself.

## Parameters

`TCanny(clip, sigma, sigma_v, t_h, t_l, mode, op, scale, opt, planes, backend)`

| Parameter | Default | Description |
|---|---|---|
| `clip` | | Constant format, 8-16 bit integer or 32-bit float. Any color family and subsampling. |
| `sigma` | `1.5` | Strength of the Gaussian blur applied first, per plane. `0` disables the blur. Chroma defaults to `sigma` divided by the subsampling factor. |
| `sigma_v` | `sigma` | Vertical blur strength, if it should differ from the horizontal one. |
| `t_h` | `8.0` | High threshold for edges (mode 0). Pixels above it start an edge. |
| `t_l` | `1.0` | Low threshold (mode 0). Pixels above it continue an edge connected to a strong pixel. Must be lower than `t_h`. |
| `mode` | `0` | `0` = edge map (white edges on black), `1` = gradient magnitude, `-1` = Gaussian blur only. |
| `op` | `1` | Gradient operator: `0` = Tritical, `1` = Prewitt, `2` = Sobel, `3` = Scharr, `4` = Kroon, `5` = Kirsch (not with `mode=0`), `6` = FDoG. |
| `scale` | `1.0` | Multiplies the gradient magnitude. |
| `opt` | `0` | CPU instruction set: `0` = best available, `1` = plain C. x86 also offers `2` = SSE2, `3` = AVX2, `4` = AVX-512. An instruction set the CPU lacks is rejected with an error. |
| `planes` | all | Planes to process. Planes left out are copied from the input. |
| `backend` | `"auto"` | `"auto"`, `"cpu"`, `"vulkan"` or `"cuda"`. `auto` uses the CPU for normal clips and Vulkan for GPU clips. A GPU backend on a normal clip, or the CPU backend on a GPU clip, is an error. |

Thresholds are given on the 8-bit scale (0-255) for every format. They are rescaled automatically for higher bit
depths and for float clips.

## Switching from TCanny r14

Replace the namespace and keep everything else:

```python
core.tcanny.TCanny(clip, ...)    # original
core.tcannyx.TCanny(clip, ...)   # this plugin, same arguments and output
```

Differences:
- **The extra `backend` argument.**
- **Unsupported `opt` values raise an error** instead of crashing.
- **Not covered:** TCanny r12 (it has the `gmmax` argument). It is an older version with different defaults and border
  handling.

## Performance

Measured at 1080p, single thread, `sigma=1.5`, milliseconds per frame (lower is better):

| CPU | Mode | TCanny r14 | tcannyx |
|---|---|---|---|
| Ryzen 9 5950X (AVX2) | edges (`mode=0`) | 17.2 | 7.8 |
| Xeon 8581C (AVX-512) | edges (`mode=0`) | 17.3 | 6.2 |
| Apple M4 Max (NEON) | edges (`mode=0`) | 34.7 (*) | 9.3 |

(*) TCanny r14 has no ARM SIMD code, so on ARM it runs plain C. The number shown is this plugin's C path (`opt=1`),
which is the same code.

At 16 threads on the 5950X, edge maps run at 0.99 ms per frame against 2.55 ms for the original.

On an RTX 5080, a 1080p 8-bit clip that stays on the GPU runs at about 4,400 frames per second with the edge map, and
9,600 with the gradient (Vulkan, 16 threads).

Tips:
- GPU backends stop scaling at about 4-8 VapourSynth threads.
- Edge maps cost more on smooth input (large `sigma`), because edges get longer to trace.
- The CPU backend uses about 27 MB of working memory per frame in flight at 1080p (about 105 MB at 4K).

## Building from source

You need Meson (>= 1.3), Ninja, a C++17 compiler and the VapourSynth Python package (`pip install "vapoursynth>=80"`),
which provides the headers.

```sh
meson setup build -Dvulkan=false -Dcuda=false   # CPU only
meson setup build -Dvulkan=true                 # CPU + Vulkan (needs Vulkan headers)
meson setup build -Dvulkan=true -Dcuda=true     # CPU + Vulkan + CUDA (needs nvcc, Linux)
meson compile -C build
meson test -C build
```

The built plugin is `build/libtcannyx.so` (`tcannyx.dll` on Windows, `libtcannyx.dylib` on macOS).

Notes:
- **Highway.** The CPU code uses [Highway](https://github.com/google/highway) 1.4 or newer for SIMD. An installed copy
  is used if found; otherwise Meson downloads it automatically. Add `--force-fallback-for=libhwy` to always use the
  bundled copy, which is what the releases do.
- **Mismatched SDKs.** If the compiler picks up a different VapourSynth SDK than your Python package, point
  `-Dvapoursynth_headers=` at the folder that contains `VSVulkan4.h`.
- **CUDA backend.** It shares memory with VapourSynth's Vulkan frames, so it needs an NVIDIA GPU whose CUDA and Vulkan
  devices match. It is Linux only for now.

## License

GPL-3.0-or-later. Based on [VapourSynth-TCanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny) by
HolyWu, a port of tcanny for AviSynth by Kevin Stone (tritical). The SIMD direction code is ported from the Vector Class
Library by Agner Fog (Apache-2.0).
