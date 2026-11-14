#!/usr/bin/env python3
"""Drop-in check against upstream TCanny (HomeOfVapourSynthEvolution/VapourSynth-TCanny, r14 or newer).

Loads this plugin (namespace tcannyx) and a built upstream plugin (namespace tcanny) in one process and checks
  1. the registered argument list is upstream's plus `backend` (and the GPU node clip type),
  2. the CPU backend is bit-identical to upstream at the same `opt` (default 1 = scalar; --opt 0 2 3 adds the SIMD paths) over
     formats x modes x operators x sigma / sigma_v x thresholds x scale x plane subsets, and
  3. with --gpu vulkan|cuda, that backend agrees with upstream's scalar output under the golden-test tolerances
     (mode 0 exact, float modes -1/1 within 2e-5, integer modes -1/1 within 1 LSB).
SIMD note: upstream r14's SIMD sources bind references to temporaries (`auto& x{ Vec4f().load_a(...) }`, 117 places), which is
undefined behaviour; GCC 15 turns it into all-zero output (MSVC and older GCC happen to work). To compare the SIMD paths, build
upstream with those lines changed to `auto x{ ... }` (no other change; this plugin's own SIMD files have no such reference).
Scalar (opt=1) is the reference otherwise, like the GPU golden tests.

usage: upstream_compare.py --plugin <this libtcannyx.so> --upstream <upstream libtcanny.so> [--gpu vulkan|cuda]
Use an empty VAPOURSYNTH_PLUGIN_PATH so nothing else autoloads."""
import argparse
import json
import sys

import numpy as np
import vapoursynth as vs

FLOAT_ATOL = 2e-5


def make_source(core, fmt, width, height):
    base = core.std.BlankClip(width=width, height=height, format=fmt, length=1)
    info = core.get_video_format(fmt)
    peak = (1 << info.bits_per_sample) - 1

    def fill(n, f):
        out = f.copy()
        for p in range(out.format.num_planes):
            arr = np.asarray(out[p])
            h, w = arr.shape
            y, x = np.indices((h, w))
            v = ((x * (13 + p) + y * (17 + 2 * p) + (x * y) % (29 + p)) % 256) / 255.0
            arr[:] = v.astype(np.float32) if info.sample_type == vs.FLOAT else np.rint(v * peak).astype(arr.dtype)
        return out

    return base.std.ModifyFrame([base], fill)


def planes_of(frame):
    return [np.asarray(frame[p]).copy() for p in range(frame.format.num_planes)]


def compare_one(core, name, source, extra, opt, failures):
    """CPU backend against upstream at the same `opt` value (the scalar case is also covered with the GPU loop below)."""
    if opt == 1:
        return  # done by the main loop (also checks that rejections agree)
    try:
        reference = core.tcanny.TCanny(source, opt=opt, **extra).get_frame(0)
    except vs.Error:
        return
    ours = core.tcannyx.TCanny(source, opt=opt, backend="cpu", **extra).get_frame(0)
    for p, (a, b) in enumerate(zip(planes_of(reference), planes_of(ours))):
        if not np.array_equal(a, b):
            failures.append((name, f"cpu opt={opt}", extra, "plane", p, "differs", int(np.count_nonzero(a != b))))
    compared[0] += 1


compared = [0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--gpu", choices=("vulkan", "cuda"))
    parser.add_argument("--opt", type=int, nargs="*", default=[1],
                        help="CPU `opt` values to compare at the same value on both sides (default: 1, scalar). Use 0 2 3 to also check the "
                             "SIMD paths; that needs an upstream build whose SIMD paths work (see the note above) and a CPU with AVX2")
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(args.upstream)
    core.std.LoadPlugin(args.plugin)
    failures = []

    # 1. interface
    if core.tcannyx.identifier != "com.mystdove.tcannyx" or core.tcannyx.namespace != "tcannyx":
        failures.append(("plugin id / namespace", core.tcannyx.identifier, core.tcannyx.namespace))
    theirs = core.tcanny.TCanny.signature.rstrip(";").split(";")
    ours = core.tcannyx.TCanny.signature.rstrip(";").split(";")
    if [a.replace("vnode:all", "vnode") for a in ours] != [a.replace("vnode:all", "vnode") for a in theirs + ["backend:data:opt"]]:
        failures.append(("signature", "upstream", theirs, "ours", ours))

    # 2/3. outputs
    cases = []
    for sigma in ([1.5], [0.0], [0.5], [4.0]):
        for mode in (-1, 0, 1):
            for op in range(7):
                if op == 5 and mode == 0:
                    continue
                cases.append(dict(sigma=sigma, mode=mode, op=op))
    cases += [dict(sigma=[1.0], sigma_v=[3.0], mode=m) for m in (-1, 0, 1)]
    cases += [dict(sigma=[2.0], sigma_v=[0.0], mode=m) for m in (-1, 0, 1)]
    cases += [dict(mode=0, t_h=30.0, t_l=10.0), dict(mode=0, t_h=200.0, t_l=100.0), dict(mode=1, scale=0.3), dict(mode=1, scale=5.1), dict(mode=0, scale=2.0),
              dict(mode=1, op=2, scale=0.25, sigma=[0.0]), dict(mode=-1), dict(), dict(mode=1)]
    formats = [("GRAY8", vs.GRAY8), ("GRAY16", vs.GRAY16), ("GRAY10", vs.GRAY10), ("GRAYS", vs.GRAYS), ("YUV420P8", vs.YUV420P8), ("YUV444P16", vs.YUV444P16),
               ("YUV420P10", vs.YUV420P10), ("YUV444PS", vs.YUV444PS), ("RGB24", vs.RGB24)]
    for name, fmt in formats:
        source = make_source(core, fmt, 200, 120)
        subsets = [None, [0], [1, 2]] if core.get_video_format(fmt).num_planes == 3 else [None]
        for kw in cases:
            for planes in subsets:
                extra = dict(kw)
                if planes is not None:
                    extra["planes"] = planes
                for opt in args.opt:
                    compare_one(core, name, source, extra, opt, failures)
                try:
                    reference = core.tcanny.TCanny(source, opt=1, **extra).get_frame(0)
                except vs.Error as exc:
                    # Upstream rejects it (plane too small, ...): ours must reject too.
                    try:
                        core.tcannyx.TCanny(source, opt=1, backend="cpu", **extra).get_frame(0)
                        failures.append((name, extra, "upstream rejects but ours accepts", str(exc)[:80]))
                    except vs.Error:
                        pass
                    continue
                for backend in ("cpu",) + ((args.gpu,) if args.gpu else ()):
                    src = source if backend == "cpu" else source.std.GPUUpload()
                    node = core.tcannyx.TCanny(src, opt=1, backend=backend, **extra)
                    try:
                        got = (node if backend == "cpu" else node.std.GPUDownload()).get_frame(0)
                    except vs.Error as exc:
                        failures.append((name, backend, extra, "error", str(exc)[:100]))
                        continue
                    compared[0] += 1
                    for p, (a, b) in enumerate(zip(planes_of(reference), planes_of(got))):
                        if backend == "cpu":
                            ok = np.array_equal(a, b)
                        elif extra.get("mode", 0) == 0:
                            ok = np.array_equal(a, b)
                        elif a.dtype.kind == "f":
                            ok = np.allclose(a, b, atol=FLOAT_ATOL, rtol=FLOAT_ATOL)
                        else:
                            ok = int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) <= 1
                        if not ok:
                            failures.append((name, backend, extra, "plane", p, "differs", int(np.count_nonzero(a != b))))
    result = {"upstream_compare": "passed" if not failures else "failed", "comparisons": compared[0], "gpu": args.gpu}
    print(json.dumps(result))
    for f in failures[:25]:
        print("FAIL", f, file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
