#!/usr/bin/env python3
"""CUDA backend vs CPU scalar (opt=1) golden comparison.

Mode 0 (binary edge mask) must match bit-exactly; modes -1/1 within 2e-5.
Covers GRAYS plus multi-plane float clips (YUV444PS, YUV420PS), blur radius
selection (sigma 0/0.1/1.5/4/10, H-only, V-only), a plane subset, and a
hysteresis stress case with a weak edge chain far longer than width+height.
"""
import argparse
import collections
import json
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
import hysteresis_corpus  # noqa: E402

ATOL = 2e-5
INT_LSB = 1  # integer modes -1/1 may differ by one rounding step


def make_source(core, fmt, width, height, painter):
    """Clip whose planes are painter(plane_index, h, w) -> array in 0..1, scaled to the format (integer formats are
    rounded to the sample range)."""
    base = core.std.BlankClip(width=width, height=height, format=fmt, length=1)
    info = core.get_video_format(fmt)
    is_float = info.sample_type == vs.FLOAT
    np_type = np.float32 if is_float else (np.uint8 if info.bytes_per_sample == 1 else np.uint16)
    peak = (1 << info.bits_per_sample) - 1

    def fill(**kwargs):
        frame = kwargs["f"].copy()
        for p in range(frame.format.num_planes):
            plane = frame[p]
            h, w = plane.shape
            array = np.asarray(plane)  # writable view; also valid for 1-pixel-wide planes
            values = np.asarray(painter(p, h, w), dtype=np.float64)
            array[:] = values.astype(np.float32) if is_float else np.rint(values * peak).astype(np_type)
        return frame

    return base.std.ModifyFrame([base], fill)


def pattern(p, h, w):
    y, x = np.indices((h, w))
    return (((x * (13 + p) + y * (17 + 2 * p) + (x * y) % (29 + p)) % 256) / 255.0).astype(np.float32)


def planes_of(frame):
    return [np.asarray(frame[p]).copy() for p in range(frame.format.num_planes)]


class Runner:
    def __init__(self, core):
        self.core = core
        self.cases = 0
        self.max_error = 0.0
        self.max_int_error = 0
        self.failures = []
        self.skipped = []
        self.edge_counts = collections.defaultdict(int)

    def compare(self, label, source, exact=False, **kw):
        mode = kw.get("mode", 0)
        try:
            cpu = self.core.tcannyx.TCanny(source, opt=1, backend="cpu", **kw).get_frame(0)
        except vs.Error as exc:
            if label.startswith("tiny-") and ("at least" in str(exc) or "too small" in str(exc)):
                self.skipped.append((label, kw, str(exc).strip()[:100]))  # front end legitimately rejects it
                return None, None
            raise
        try:
            cuda = self.core.tcannyx.TCanny(source.std.GPUUpload(), backend="cuda", **kw).std.GPUDownload().get_frame(0)
        except vs.Error as exc:
            self.failures.append((label, "error", kw, str(exc)[:160]))
            self.cases += 1
            return cpu, None
        for p, (a, b) in enumerate(zip(planes_of(cpu), planes_of(cuda))):
            integer = a.dtype.kind in "ui"
            error = float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if integer else float(np.max(np.abs(a - b)))
            if integer:
                self.max_int_error = max(self.max_int_error, int(error))
            else:
                self.max_error = max(self.max_error, error)
            if mode == 0:
                self.edge_counts[label] += int(np.count_nonzero(a))
                ok = np.array_equal(a, b)
            elif exact:
                ok = np.array_equal(a, b)
            elif integer:
                ok = error <= INT_LSB
            else:
                ok = np.allclose(a, b, atol=ATOL, rtol=ATOL)
            if not ok:
                self.failures.append((label, p, kw, error, int(np.count_nonzero(a != b))))
        self.cases += 1
        return cpu, cuda


def valid_combos():
    for op in range(7):
        for mode in (-1, 0, 1):
            if op == 5 and mode == 0:
                continue
            yield op, mode


# t_h/t_l are given in 0..255 units (the plugin divides by 255 for float input).
# Chosen so mode 0 of the test pattern yields a non-trivial mask with real weak pixels.
TH = dict(t_h=153.0, t_l=51.0)


def serpentine(h, w, period=6, thick=3, strong=1.0, weak=0.2):
    """A snake-shaped plateau (weak value) that starts in a strong blob."""
    img = np.zeros((h, w), dtype=np.float32)
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
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(args.plugin)
    r = Runner(core)
    width, height = 128, 96

    gray = make_source(core, vs.GRAYS, width, height, pattern)

    # 1. GRAYS: every sigma x valid (op, mode)
    for sigma in (0.0, 0.1, 1.5, 4.0):
        for op, mode in valid_combos():
            r.compare("gray-sigma", gray, sigma=[sigma], mode=mode, op=op, **TH)

    # 1b. scale != 1: components are scaled before the magnitude, so every mode must match the CPU exactly
    for scale in (0.3, 1.7, 3.3):
        for sigma in (0.0, 1.5):
            for op, mode in valid_combos():
                r.compare("gray-scale", gray, exact=True, sigma=[sigma], mode=mode, op=op, scale=scale, **TH)

    # 1c. integer formats (the original TCanny's 8-16 bit support): every valid combo for sigma 0 / 1.5, and subsampled YUV
    for name, fmt in (("GRAY8", vs.GRAY8), ("GRAY16", vs.GRAY16), ("GRAY10", vs.GRAY10), ("GRAY12", vs.GRAY12)):
        clip = make_source(core, fmt, width, height, pattern)
        for sigma in (0.0, 1.5):
            for op, mode in valid_combos():
                r.compare(f"{name}-sigma", clip, sigma=[sigma], mode=mode, op=op, **TH)
        for scale in (0.3, 1.7):
            for op, mode in valid_combos():
                r.compare(f"{name}-scale", clip, sigma=[1.5], mode=mode, op=op, scale=scale, **TH)
        for sigma_h, sigma_v in ((1.5, 0.0), (0.0, 1.5), (4.0, 0.0), (1.0, 3.0)):
            for op, mode in ((1, -1), (1, 0), (1, 1), (6, 0), (6, 1), (5, 1)):
                r.compare(f"{name}-hv", clip, sigma=[sigma_h], sigma_v=[sigma_v], mode=mode, op=op, **TH)
    for name, fmt in (("YUV420P8", vs.YUV420P8), ("YUV444P16", vs.YUV444P16), ("YUV420P10", vs.YUV420P10), ("RGB24", vs.RGB24)):
        clip = make_source(core, fmt, width, height, pattern)
        for sigma in (0.0, 1.5):
            for op, mode in valid_combos():
                r.compare(name, clip, sigma=[sigma], mode=mode, op=op, **TH)
        for planes in ([0], [1], [1, 2]):
            for mode in (-1, 0, 1):
                r.compare(f"{name}-planes{planes}", clip, planes=planes, sigma=[1.5], mode=mode, op=2, **TH)

    # 2. H-only / V-only blur (sigma_v=0 / sigma=0), and a large radius (sigma 10 -> r=30)
    for sigma_h, sigma_v in ((1.5, 0.0), (0.0, 1.5), (4.0, 0.0), (0.0, 4.0), (0.1, 4.0), (4.0, 0.1), (1.0, 3.0)):
        for op, mode in ((1, -1), (1, 0), (1, 1), (2, 0), (6, 0), (6, 1), (5, 1)):
            r.compare("gray-hv", gray, sigma=[sigma_h], sigma_v=[sigma_v], mode=mode, op=op, **TH)
    big = make_source(core, vs.GRAYS, 160, 128, pattern)
    for sigma_h, sigma_v in ((10.0, 10.0), (10.0, 0.0), (0.0, 10.0)):
        for mode in (-1, 0, 1):
            r.compare("gray-sigma10", big, sigma=[sigma_h], sigma_v=[sigma_v], mode=mode, op=1, t_h=1.0, t_l=0.3)

    # 3. Multi-plane float clips
    for fmt_name in ("YUV444PS", "YUV420PS"):
        fmt = getattr(vs, fmt_name)
        clip = make_source(core, fmt, width, height, pattern)
        for sigma in (0.0, 1.5):
            for op, mode in valid_combos():
                r.compare(fmt_name, clip, sigma=[sigma], mode=mode, op=op, **TH)
        # plane subsets: unprocessed planes must come through untouched
        for planes in ([0], [1], [1, 2]):
            for mode in (-1, 0, 1):
                cpu, cuda = r.compare(f"{fmt_name}-planes{planes}", clip, planes=planes, sigma=[1.5], mode=mode, op=2, **TH)
                src = planes_of(clip.get_frame(0))
                for p, out in enumerate(planes_of(cuda) if cuda is not None else []):
                    if p not in planes and not np.array_equal(out, src[p]):
                        r.failures.append((fmt_name, p, "unprocessed plane modified", planes, mode))

    # 4. Hysteresis stress: a weak chain much longer than width + height
    sh, sw = 256, 256
    snake = serpentine(sh, sw)
    stress = make_source(core, vs.GRAYS, sw, sh, lambda p, h, w: snake)
    kw = dict(sigma=[0.0], mode=0, op=1, t_h=255.0, t_l=25.5)
    cpu, cuda = r.compare("hysteresis-stress", stress, **kw)
    if cuda is None:
        print("stress: CUDA run errored", file=sys.stderr)
    edges = np.asarray(cpu[0]) != 0
    seeds = np.zeros_like(edges)
    seeds[0:20, 0:14] = True
    far, reached = bfs_max_distance(edges, seeds)
    total = int(np.count_nonzero(edges))
    print(json.dumps({"stress_edges": total, "stress_chain_bfs_distance": far, "stress_reached": reached,
                      "width_plus_height": sw + sh}))
    if far <= 10 * (sw + sh) or reached < 0.9 * total:
        r.failures.append(("hysteresis-stress input does not exercise a long chain", far, reached, total))
    # Without promotion, strong-only seeds are tiny: check weak pixels exist that only hysteresis can reach.
    weak_only = core.tcannyx.TCanny(stress, sigma=[0.0], mode=0, op=1, t_h=255.0, t_l=254.9, opt=1, backend="cpu").get_frame(0)
    if np.count_nonzero(np.asarray(weak_only[0])) >= total // 4:
        r.failures.append(("stress: strong-only edge count too large", int(np.count_nonzero(np.asarray(weak_only[0]))), total))

    # 5. Tiny planes. Mode -1 only requires dimension > blur radius, so 1- and 2-pixel
    # dimensions are legal and must mirror exactly like the CPU (x=-1 -> 1, x=2 -> 0).
    for (tw, th), sig in (((2, 17), [(0.3, 0.0), (0.3, 0.3), (0.0, 0.0)]),
                          ((17, 2), [(0.0, 0.3), (0.3, 0.3), (0.0, 0.0)]),
                          ((1, 17), [(0.0, 0.0), (0.0, 1.0), (0.0, 0.3)]),
                          ((17, 1), [(0.0, 0.0), (1.0, 0.0), (0.3, 0.0)]),
                          ((2, 2), [(0.3, 0.3), (0.0, 0.0)]),
                          ((1, 1), [(0.0, 0.0)])):
        tiny = make_source(core, vs.GRAYS, tw, th, pattern)
        for sh_, sv_ in sig:
            for op in (0, 1, 2, 3, 4, 5, 6):
                r.compare(f"tiny-{tw}x{th}", tiny, sigma=[sh_], sigma_v=[sv_], mode=-1, op=op)
    tiny3 = make_source(core, vs.GRAYS, 3, 3, pattern)
    for sigma in (0.0,):
        for op, mode in valid_combos():
            if mode != -1:
                r.compare("tiny-3x3", tiny3, sigma=[sigma], sigma_v=[sigma], mode=mode, op=op, t_h=60.0, t_l=20.0)
    for op in (0, 1, 2, 3, 4, 5):  # FDOG needs >= 3 and radius 2 reach; covered above
        r.compare("tiny-3x3-mode-1", tiny3, sigma=[0.0], sigma_v=[0.0], mode=-1, op=op)

    # Several frames in flight on one instance, each with a different chain depth (and so a different hysteresis batch
    # count while the instance's depth hint adapts): all must match the CPU.
    sh, sw, count = 192, 192, 24
    snake = serpentine(sh, sw)
    base = core.std.BlankClip(width=sw, height=sh, format=vs.GRAYS, length=count)

    def fill_n(n, f):
        frame = f.copy()
        rows = max(8, (n * 7 % count + 1) * sh // count)  # chain length varies non-monotonically between frames
        image = snake.copy()
        image[rows:, :] = 0.0
        np.asarray(frame[0])[:] = image.astype(np.float32)
        return frame

    multi = base.std.ModifyFrame([base], fill_n)
    kw_multi = dict(sigma=[0.0], mode=0, op=1, t_h=255.0, t_l=25.5)
    cpu_frames = [np.asarray(f[0]).copy() for f in core.tcannyx.TCanny(multi, opt=1, backend="cpu", **kw_multi).frames(prefetch=1)]
    for n, f in enumerate(core.tcannyx.TCanny(multi.std.GPUUpload(), backend="cuda", **kw_multi).std.GPUDownload().frames(prefetch=8)):
        if not np.array_equal(cpu_frames[n], np.asarray(f[0])):
            r.failures.append(("varying-depth-frames", n, int(np.count_nonzero(cpu_frames[n] != np.asarray(f[0])))))
    r.cases += count

    corpus_stats = collections.Counter()
    hysteresis_corpus.run(r.compare, lambda fmt, w, h, painter: make_source(core, fmt, w, h, painter),
                          [("GRAYS", vs.GRAYS), ("GRAY8", vs.GRAY8), ("GRAY16", vs.GRAY16)], hysteresis_corpus.cases, r.failures, corpus_stats)
    hysteresis_corpus.run(r.compare, lambda fmt, w, h, painter: make_source(core, fmt, w, h, painter),
                          [("YUV444PS", vs.YUV444PS), ("YUV420P8", vs.YUV420P8)], hysteresis_corpus.yuv420_cases, r.failures, corpus_stats)
    print(json.dumps({"hysteresis_corpus": dict(corpus_stats)}))
    for label in [k for k in r.edge_counts if k.startswith("hyst-")]:
        del r.edge_counts[label]  # negative cases are empty by design; the corpus checks its own non-vacuity
    if corpus_stats["chain_nonempty"] < 0.5 * corpus_stats["chain_cases"]:
        r.failures.append(("hysteresis corpus is mostly vacuous", dict(corpus_stats)))

    nontrivial = {k: v for k, v in r.edge_counts.items()}
    summary = {"backend": "cuda", "status": "passed" if not r.failures else "failed", "cases": r.cases,
               "max_abs_error": r.max_error, "max_int_error_lsb": r.max_int_error, "skipped_by_front_end": len(r.skipped), "mode0_nonzero_pixels": nontrivial}
    print(json.dumps(summary))
    if r.failures:
        print("failures by group:", dict(collections.Counter(str(f[0]) for f in r.failures)), file=sys.stderr)
        for f in r.failures[:30]:
            print("FAIL", f, file=sys.stderr)
        print(f"{len(r.failures)} failures", file=sys.stderr)
        raise SystemExit(1)
    if any(v == 0 for v in r.edge_counts.values()):
        print("warning: some mode 0 groups had no edges at all", file=sys.stderr)


if __name__ == "__main__":
    main()
