#!/usr/bin/env python3
"""Vulkan backend vs CPU scalar (opt=1) golden comparison.

The CPU scalar path is the oracle (never opt=0: the SIMD paths differ from it on binary
tie-breaks). Comparison rules:

* mode 0 (binary edge mask): bit-exact for every format.
* mode -1 / 1, float: |gpu - cpu| <= 2e-5 (absolute and relative).
* mode -1 / 1, integer: |gpu - cpu| <= 1 LSB. The GPU and CPU compute the float blur/gradient
  with different instruction selection, so a value that lands within a few float ULPs of an
  x.5 rounding boundary can round to the neighbouring integer; the number of such pixels is
  printed so a regression to "many off-by-one" is visible.

Coverage: GRAYS / GRAY8 / GRAY16 over sigma {0, 0.1, 1.5, 4.0} x every valid (op, mode); H-only and
V-only blur; sigma 10 (radius 30); odd plane sizes; a scale != 1 case; YUV420P8 (chroma planes are blurred with
sigma/2) incl. plane subsets; the auto/cpu backend selection checks; and hysteresis stress
cases whose weak chain is far longer than width + height (all three sample formats).
"""
import argparse
import collections
import json
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
import hysteresis_corpus  # noqa: E402

FLOAT_ATOL = 2e-5

FORMATS = {
    "GRAY8": (vs.GRAY8, np.uint8, 255),
    "GRAY16": (vs.GRAY16, np.uint16, 65535),
    "GRAYS": (vs.GRAYS, np.float32, None),
}


def make_source(core, fmt, width, height, painter):
    """Clip whose planes are painter(plane_index, h, w) -> float array in 0..1 (scaled to the format)."""
    base = core.std.BlankClip(width=width, height=height, format=fmt, length=1)
    info = core.get_video_format(fmt)
    sample_type = info.sample_type
    bits = info.bits_per_sample
    np_type = np.float32 if sample_type == vs.FLOAT else (np.uint8 if info.bytes_per_sample == 1 else np.uint16)
    peak = (1 << bits) - 1

    def fill(**kwargs):
        frame = kwargs["f"].copy()
        for p in range(frame.format.num_planes):
            plane = frame[p]
            h, w = plane.shape
            array = np.asarray(plane)  # writable (frame was copied), honours the row stride
            values = np.asarray(painter(p, h, w), dtype=np.float64)
            if sample_type == vs.FLOAT:
                array[:] = values.astype(np.float32)
            else:
                array[:] = np.rint(values * peak).astype(np_type)
        return frame

    return base.std.ModifyFrame([base], fill)


def pattern(p, h, w):
    y, x = np.indices((h, w))
    return (((x * (13 + p) + y * (17 + 2 * p) + (x * y) % (29 + p)) % 256) / 255.0)


def planes_of(frame):
    return [np.asarray(frame[p]).copy() for p in range(frame.format.num_planes)]


class Runner:
    def __init__(self, core):
        self.core = core
        self.cases = 0
        self.failures = []
        self.max_float_error = 0.0
        self.max_int_error = 0
        self.int_off_by_one = 0
        self.edge_counts = collections.defaultdict(int)
        self.mode0_cases = 0

    def compare(self, label, source, **kw):
        mode = kw.get("mode", 0)
        self.cases += 1
        cpu = self.core.tcannyx.TCanny(source, opt=1, backend="cpu", **kw).get_frame(0)
        try:
            gpu = self.core.tcannyx.TCanny(source.std.GPUUpload(), backend="vulkan", **kw).std.GPUDownload().get_frame(0)
        except vs.Error as exc:
            self.failures.append((label, "error", kw, str(exc)[:200]))
            return cpu, None
        is_float = cpu.format.sample_type == vs.FLOAT
        for p, (a, b) in enumerate(zip(planes_of(cpu), planes_of(gpu))):
            if mode == 0:
                self.mode0_cases += 1
                self.edge_counts[label] += int(np.count_nonzero(a))
                if not np.array_equal(a, b):
                    self.failures.append((label, p, kw, "mode0 mismatches", int(np.count_nonzero(a != b))))
            elif is_float:
                error = float(np.max(np.abs(a - b)))
                self.max_float_error = max(self.max_float_error, error)
                if not np.allclose(a, b, atol=FLOAT_ATOL, rtol=FLOAT_ATOL):
                    self.failures.append((label, p, kw, "float error", error, int(np.count_nonzero(np.abs(a - b) > FLOAT_ATOL))))
            else:
                diff = np.abs(a.astype(np.int64) - b.astype(np.int64))
                error = int(diff.max())
                self.max_int_error = max(self.max_int_error, error)
                self.int_off_by_one += int(np.count_nonzero(diff))
                if error > 1:
                    self.failures.append((label, p, kw, "integer error > 1 LSB", error, int(np.count_nonzero(diff > 1))))
        return cpu, gpu


def is_size_error(exc):
    text = str(exc).lower()
    return any(k in text for k in ("at least", "radius", "too small", "smaller")) and any(k in text for k in ("height", "width", "plane", "size", "radius"))


def valid_combos():
    for op in range(7):
        for mode in (-1, 0, 1):
            if op == 5 and mode == 0:
                continue
            yield op, mode


# t_h/t_l are given in 0..255 units (the plugin rescales them for the sample format).
# Chosen so mode 0 of the test pattern yields a non-trivial mask with real weak pixels.
TH = dict(t_h=153.0, t_l=51.0)


def serpentine(h, w, period=6, thick=3, strong=1.0, weak=0.2):
    """A snake-shaped plateau (weak value) that starts in a strong blob."""
    img = np.zeros((h, w), dtype=np.float64)
    margin = 4
    rows = list(range(margin, h - margin - thick + 1, period))
    for i, r in enumerate(rows):
        img[r:r + thick, margin:w - margin] = weak
        if i + 1 < len(rows):  # vertical connector, alternating ends
            c = (w - margin - thick) if i % 2 == 0 else margin
            img[r:rows[i + 1] + thick, c:c + thick] = weak
    img[rows[0] - 2:rows[0] + thick + 2, margin - 3:margin + 8] = strong  # strong blob at the head
    return img


def bfs_max_distance(edges, seeds):
    h, w = edges.shape
    dist = np.full((h, w), -1, dtype=np.int64)
    queue = collections.deque()
    for y, x in zip(*np.nonzero(seeds & edges)):
        dist[y, x] = 0
        queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and edges[yy, xx] and dist[yy, xx] < 0:
                    dist[yy, xx] = dist[y, x] + 1
                    queue.append((yy, xx))
    return int(dist.max()), int(np.count_nonzero(dist >= 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--strict", action="store_true",
                        help="accepted for compatibility; every comparison is strict now (no warning downgrade)")
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(args.plugin)
    r = Runner(core)
    width, height = 128, 96
    clips = {name: make_source(core, fmt, width, height, pattern) for name, (fmt, _, _) in FORMATS.items()}

    # 1. every sigma x valid (op, mode), for every gray format
    for name, clip in clips.items():
        for sigma in (0.0, 0.1, 1.5, 4.0):
            for op, mode in valid_combos():
                r.compare(f"{name}-sigma", clip, sigma=[sigma], mode=mode, op=op, **TH)

    # 2. H-only / V-only blur (sigma_v=0 / sigma=0), mixed radii, scale != 1
    for name, clip in clips.items():
        for sigma_h, sigma_v in ((1.5, 0.0), (0.0, 1.5), (4.0, 0.0), (0.0, 4.0), (0.1, 4.0), (4.0, 0.1), (1.0, 3.0)):
            for op, mode in ((1, -1), (1, 0), (1, 1), (2, 0), (6, 0), (6, 1), (5, 1)):
                r.compare(f"{name}-hv", clip, sigma=[sigma_h], sigma_v=[sigma_v], mode=mode, op=op, **TH)
        for sigma in (0.0, 1.5):
            for op, mode in valid_combos():
                for scale in (0.3, 1.7, 3.3):
                    r.compare(f"{name}-scale", clip, sigma=[sigma], mode=mode, op=op, scale=scale, **TH)
    # sigma 10 -> radius 30, and a big plane
    for name, (fmt, _, _) in FORMATS.items():
        big = make_source(core, fmt, 160, 128, pattern)
        for sigma_h, sigma_v in ((10.0, 10.0), (10.0, 0.0), (0.0, 10.0)):
            for mode in (-1, 0, 1):
                r.compare(f"{name}-sigma10", big, sigma=[sigma_h], sigma_v=[sigma_v], mode=mode, op=1, t_h=1.0, t_l=0.3)

    # 3. YUV420P8: chroma is 64x48 and defaults to sigma/2; plane subsets must leave others untouched
    yuv = make_source(core, vs.YUV420P8, width, height, pattern)
    for sigma in (0.0, 1.5, 4.0):
        for op, mode in valid_combos():
            r.compare("YUV420P8", yuv, sigma=[sigma], mode=mode, op=op, **TH)
    for planes in ([0], [1], [1, 2]):
        for mode in (-1, 0, 1):
            cpu, gpu = r.compare(f"YUV420P8-planes{planes}", yuv, planes=planes, sigma=[1.5], mode=mode, op=2, **TH)
            src = planes_of(yuv.get_frame(0))
            for p, out in enumerate(planes_of(gpu) if gpu is not None else []):
                if p not in planes and not np.array_equal(out, src[p]):
                    r.failures.append(("YUV420P8", p, "unprocessed plane modified", planes, mode))

    # 3b. odd plane sizes: partial workgroups, padded strides, planes barely larger than the stencil
    for name in ("GRAY8", "GRAYS"):
        fmt = FORMATS[name][0]
        for ow, oh in ((61, 37), (33, 17), (7, 9)):
            odd = make_source(core, fmt, ow, oh, pattern)
            for sigma in (0.0, 0.6, 1.0):
                for op, mode in valid_combos():
                    r.compare(f"{name}-odd-{ow}x{oh}", odd, sigma=[sigma], mode=mode, op=op, **TH)

    # 3c. tiny planes. Mode -1 is only limited by the blur radius, so 1- and 2-pixel dimensions are legal and
    # must match the CPU (2-pixel dimensions mirror: -1 -> 1, 2 -> 0). 3x3 is the minimum for modes 0/1.
    # A case is skipped only when creation fails with the front end's size rejection; skips are counted.
    skipped = 0
    for name in ("GRAY8", "GRAY16", "GRAYS"):
        fmt = FORMATS[name][0]
        tiny = []
        for n in (2, 3, 5, 17, 40):
            tiny.append((2, n, 0.3, 0.0))   # 2xN: radius 1 horizontally, none vertically
            tiny.append((n, 2, 0.0, 0.3))   # Nx2: radius 1 vertically
            tiny.append((2, n, 0.3, 0.3))
        for n in (1, 2, 5, 17):
            tiny.append((1, n, 0.0, 0.0))
            tiny.append((n, 1, 0.0, 0.0))
        tiny += [(2, 2, 0.0, 0.0), (1, 1, 0.0, 0.0)]
        for tw, th, sh, sv in tiny:
            clip = make_source(core, fmt, tw, th, pattern)
            before = r.cases
            try:
                r.compare(f"{name}-tiny-{tw}x{th}", clip, sigma=[sh], sigma_v=[sv], mode=-1, op=1)
            except vs.Error as exc:
                if is_size_error(exc):
                    skipped += 1
                    r.cases = before
                    continue
                raise
        clip = make_source(core, fmt, 3, 3, pattern)
        for op, mode in valid_combos():
            if mode == -1:
                continue
            try:
                r.compare(f"{name}-3x3", clip, sigma=[0.0], mode=mode, op=op, **TH)
            except vs.Error as exc:
                if not is_size_error(exc):
                    raise
                skipped += 1
                r.cases -= 1
    print(json.dumps({"tiny_cases_skipped_by_front_end": skipped}))

    # 4. backend selection: auto resolves to Vulkan for a GPU node, cpu refuses a GPU node
    uploaded = clips["GRAYS"].std.GPUUpload()
    for mode in (-1, 0, 1):
        vk = core.tcannyx.TCanny(uploaded, sigma=[1.5], mode=mode, op=1, backend="vulkan").std.GPUDownload().get_frame(0)
        auto = core.tcannyx.TCanny(uploaded, sigma=[1.5], mode=mode, op=1, backend="auto").std.GPUDownload().get_frame(0)
        if not np.array_equal(np.asarray(vk[0]), np.asarray(auto[0])):
            r.failures.append(("auto differs from vulkan", mode))
        try:
            core.tcannyx.TCanny(uploaded, mode=mode, backend="cpu")
        except vs.Error:
            pass
        else:
            r.failures.append(("backend=cpu accepted a GPU node", mode))

    # 5. Hysteresis stress: a weak chain much longer than width + height, in every sample format
    stress_info = {}
    for name, (fmt, _, _) in FORMATS.items():
        for sh, sw in ((256, 256), (200, 96)):
            snake = serpentine(sh, sw)
            stress = make_source(core, fmt, sw, sh, lambda p, h, w: snake)
            kw = dict(sigma=[0.0], mode=0, op=1, t_h=255.0, t_l=25.5)
            cpu, _ = r.compare(f"hysteresis-stress-{name}-{sw}x{sh}", stress, **kw)
            edges = np.asarray(cpu[0]) != 0
            seeds = np.zeros_like(edges)
            seeds[0:20, 0:14] = True
            far, reached = bfs_max_distance(edges, seeds)
            total = int(np.count_nonzero(edges))
            stress_info[f"{name}-{sw}x{sh}"] = {"edges": total, "bfs_chain": far, "reached": reached, "w+h": sw + sh}
            # The input must really exercise the long chain: the BFS distance from the strong blob to the far
            # end of the CPU edge set is several times width+height, and nearly all edge pixels hang off it.
            if far <= 5 * (sw + sh) or reached < 0.9 * total:
                r.failures.append(("hysteresis-stress input does not exercise a long chain", name, sw, sh, far, reached, total))
            # Strong-only (t_l just under t_h) must keep far fewer edges: the rest is only reachable via hysteresis.
            strong_only = core.tcannyx.TCanny(stress, sigma=[0.0], mode=0, op=1, t_h=255.0, t_l=254.9, opt=1, backend="cpu").get_frame(0)
            if np.count_nonzero(np.asarray(strong_only[0])) >= total // 4:
                r.failures.append(("stress: strong-only edge count too large", name,
                                   int(np.count_nonzero(np.asarray(strong_only[0]))), total))
    print(json.dumps({"hysteresis_stress": stress_info}))

    # 5b. Adversarial hysteresis corpus (word/tile boundaries, padding bits, seeds at the edges), exact vs CPU scalar.
    corpus_stats = collections.Counter()
    hysteresis_corpus.run(r.compare, lambda fmt, w, h, painter: make_source(core, fmt, w, h, painter),
                          [(n, f) for n, (f, _, _) in FORMATS.items()], hysteresis_corpus.cases, r.failures, corpus_stats)
    hysteresis_corpus.run(r.compare, lambda fmt, w, h, painter: make_source(core, fmt, w, h, painter),
                          [("YUV420P8", vs.YUV420P8)], hysteresis_corpus.yuv420_cases, r.failures, corpus_stats)
    print(json.dumps({"hysteresis_corpus": dict(corpus_stats)}))
    for label in [k for k in r.edge_counts if k.startswith("hyst-")]:
        del r.edge_counts[label]  # negative cases are empty by design; the corpus checks its own non-vacuity
    if corpus_stats["chain_nonempty"] < 0.5 * corpus_stats["chain_cases"]:
        r.failures.append(("hysteresis corpus is mostly vacuous", dict(corpus_stats)))

    # 6. concurrent frames: several frames in flight on the same filter instance (shared exec pool), each
    # needing a different number of hysteresis batches, must all match the CPU.
    sh, sw, count = 192, 192, 24
    snake = serpentine(sh, sw)
    base = core.std.BlankClip(width=sw, height=sh, format=vs.GRAY8, length=count)

    def fill_n(n, f):
        frame = f.copy()
        # frame n keeps only the first (n+1)/count of the serpentine rows, so chain lengths differ per frame
        rows = max(8, (n + 1) * sh // count)
        image = snake.copy()
        image[rows:, :] = 0.0
        np.asarray(frame[0])[:] = np.rint(image * 255).astype(np.uint8)
        return frame

    multi = base.std.ModifyFrame([base], fill_n)
    kw = dict(sigma=[0.0], mode=0, op=1, t_h=255.0, t_l=25.5)
    cpu_clip = core.tcannyx.TCanny(multi, opt=1, backend="cpu", **kw)
    gpu_clip = core.tcannyx.TCanny(multi.std.GPUUpload(), backend="vulkan", **kw).std.GPUDownload()
    cpu_frames = [np.asarray(f[0]).copy() for f in cpu_clip.frames(prefetch=1)]
    for n, f in enumerate(gpu_clip.frames(prefetch=8)):
        if not np.array_equal(cpu_frames[n], np.asarray(f[0])):
            r.failures.append(("concurrent-frames", n, int(np.count_nonzero(cpu_frames[n] != np.asarray(f[0])))))
    r.cases += count

    summary = {"backend": "vulkan", "status": "passed" if not r.failures else "failed", "cases": r.cases,
               "mode0_plane_comparisons": r.mode0_cases, "max_float_abs_error": r.max_float_error,
               "max_int_error_lsb": r.max_int_error, "int_pixels_off_by_one": r.int_off_by_one}
    print(json.dumps(summary))
    if r.failures:
        for f in r.failures[:40]:
            print("FAIL", f, file=sys.stderr)
        by_label = collections.Counter(str(f[0]) for f in r.failures)
        print("failures by group:", dict(by_label), file=sys.stderr)
        print(f"{len(r.failures)} failures", file=sys.stderr)
        raise SystemExit(1)
    empty = [k for k, v in r.edge_counts.items() if v == 0]
    if empty:
        print("FAIL mode 0 groups with no edges at all (test would be vacuous):", empty, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
