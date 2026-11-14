"""Adversarial hysteresis inputs shared by vulkan_golden.py and cuda_golden.py (docs/perf-plan.md, B0).

Every case is a float image in 0..1: background 0, a weak plateau (0.2) drawn as a thick polyline, and a strong blob
(1.0) at a chosen end of it. With sigma 0, op 1 and t_h=255, t_l=25.5 the weak outline survives NMS and only hysteresis
connects it to the strong blob, so the output depends on the exact propagation. The cases target the places a
bit-packed implementation (32 pixels per word, tiles of 32 rows) can go wrong: word and tile boundaries, the padding
bits of the last word, the last row and column, and chains that leave and re-enter a word or tile.
"""
import numpy as np

KW = dict(sigma=[0.0], mode=0, op=1, t_h=255.0, t_l=25.5)
WEAK = 0.2
STRONG = 1.0
WIDTHS = (3, 31, 32, 33, 63, 64, 65, 127, 129)
HEIGHTS = (3, 31, 32, 33, 34, 65)
LARGE = (1920, 1080)
SEEDS = ("start", "middle", "end", "last_col", "last_row", "padding")
THICK = 3


def _stamp(img, y, x, t=THICK, value=WEAK):
    h, w = img.shape
    y0, x0 = max(0, y - t // 2), max(0, x - t // 2)
    img[y0:min(h, y0 + t), x0:min(w, x0 + t)] = value


def polyline(img, points, value=WEAK):
    """Draw a thick polyline through integer (y, x) points; returns the list of visited centre points."""
    path = []
    for (y0, x0), (y1, x1) in zip(points[:-1], points[1:]):
        n = max(abs(y1 - y0), abs(x1 - x0)) + 1
        ys = np.rint(np.linspace(y0, y1, n)).astype(int)
        xs = np.rint(np.linspace(x0, x1, n)).astype(int)
        for y, x in zip(ys, xs):
            _stamp(img, y, x, value=value)
            path.append((int(y), int(x)))
    return path


def _clamp(v, hi):
    return int(min(max(v, 0), hi - 1))


def shape_points(name, h, w):
    """Centre-line vertices of a named chain shape for an h x w plane."""
    m = 1
    if name == "hline":
        return [(h // 2, m), (h // 2, w - 1 - m)]
    if name == "vline":
        return [(m, w // 2), (h - 1 - m, w // 2)]
    if name == "diag":
        return [(m, m), (h - 1 - m, w - 1 - m)]
    if name == "antidiag":
        return [(m, w - 1 - m), (h - 1 - m, m)]
    if name == "serp_h":  # rows connected alternately at the right and left end
        pts, step = [], 6
        rows = list(range(m, h - m, step)) or [m]
        for i, r in enumerate(rows):
            a, b = (m, w - 1 - m) if i % 2 == 0 else (w - 1 - m, m)
            pts += [(r, a), (r, b)]
        return pts
    if name == "serp_v":
        pts, step = [], 6
        cols = list(range(m, w - m, step)) or [m]
        for i, c in enumerate(cols):
            a, b = (m, h - 1 - m) if i % 2 == 0 else (h - 1 - m, m)
            pts += [(a, c), (b, c)]
        return pts
    if name == "spiral":  # inward rectangular spiral
        pts, (t, l, b, r) = [], (m, m, h - 1 - m, w - 1 - m)
        while t <= b and l <= r:
            pts += [(t, l), (t, r), (b, r), (b, l + 4)]
            t, l, b, r = t + 6, l + 6, b - 6, r - 6
            if t <= b and l <= r:
                pts.append((t, l - 2 if l - 2 >= 0 else 0))
        return pts
    if name == "uturn":  # leaves a 32-row tile and a 32-column word and comes back through the same one
        c = _clamp(31, w)
        top, bottom = _clamp(2, h), _clamp(h - 3, h)
        mid = _clamp(33, h)
        return [(top, c - 1 if c > 1 else 0), (bottom, c - 1 if c > 1 else 0), (bottom, _clamp(c + 4, w)), (mid, _clamp(c + 4, w)),
                (mid, _clamp(c + 1, w)), (top, _clamp(c + 1, w))]
    raise KeyError(name)


def _seed_blob(img, path, seed, h, w):
    """Put the strong blob on the chain at the requested place."""
    if seed == "start":
        y, x = path[0]
    elif seed == "end":
        y, x = path[-1]
    elif seed == "middle":
        y, x = path[len(path) // 2]
    elif seed == "last_col":
        y, x = min(path, key=lambda p: (-p[1], p[0]))
    elif seed == "last_row":
        y, x = min(path, key=lambda p: (-p[0], p[1]))
    elif seed == "padding":  # the pixel x = w - 1 when w % 32 != 0 sits next to the padding bits of the last word
        y, x = min(path, key=lambda p: (abs(p[1] - (w - 1)), p[0]))
    else:
        raise KeyError(seed)
    _stamp(img, y, x, t=THICK, value=STRONG)


def make_image(name, h, w, seed, broken=False, with_seed=True):
    img = np.zeros((h, w), dtype=np.float64)
    pts = shape_points(name, h, w)
    path = polyline(img, pts)
    if broken:  # a single pixel gap, wide enough to cut the 3-thick outline in the middle of the chain
        y, x = path[len(path) // 2]
        img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 0.0
    if with_seed:
        _seed_blob(img, path, seed, h, w)
    return img


SHAPES = ("hline", "vline", "diag", "antidiag", "serp_h", "serp_v", "spiral", "uturn")


def merge_image(h, w):
    """Two chains that meet: a 'Y' whose arms start at different strong seeds."""
    img = np.zeros((h, w), dtype=np.float64)
    a = polyline(img, [(1, 1), (h // 2, w // 2)])
    b = polyline(img, [(1, w - 2), (h // 2, w // 2)])
    polyline(img, [(h // 2, w // 2), (h - 2, w // 2)])
    _stamp(img, *a[0], value=STRONG)
    _stamp(img, *b[0], value=STRONG)
    return img


def cases(include_large=True):
    """Yield (label, width, height, painter, kind); painter(plane, h, w) -> float image, kind in {'chain', 'negative'}."""
    counter = 0
    for h in HEIGHTS:
        for w in WIDTHS:
            for name in SHAPES:
                seed = SEEDS[counter % len(SEEDS)]
                counter += 1
                yield (f"hyst-{name}-{w}x{h}-{seed}", w, h,
                       lambda p, hh, ww, name=name, seed=seed: make_image(name, hh, ww, seed), "chain")
            yield f"hyst-merge-{w}x{h}", w, h, lambda p, hh, ww: merge_image(hh, ww), "chain"
            yield (f"hyst-noseed-{w}x{h}", w, h,
                   lambda p, hh, ww: make_image("serp_h", hh, ww, "start", with_seed=False), "negative")
            yield (f"hyst-broken-{w}x{h}", w, h,
                   lambda p, hh, ww: make_image("hline", hh, ww, "start", broken=True), "negative")
    if include_large:
        w, h = LARGE
        for name, seed in (("serp_h", "start"), ("spiral", "padding"), ("hline", "last_col"), ("uturn", "end")):
            yield (f"hyst-{name}-{w}x{h}-{seed}", w, h,
                   lambda p, hh, ww, name=name, seed=seed: make_image(name, hh, ww, seed), "chain")


def yuv420_cases():
    """Even sizes for subsampled formats (chroma planes are half size)."""
    for (w, h) in ((64, 34), (66, 34), (130, 66), (34, 64), (96, 98), (64, 64)):
        for k, name in enumerate(SHAPES):
            seed = SEEDS[k % len(SEEDS)]
            yield (f"hyst-yuv-{name}-{w}x{h}-{seed}", w, h,
                   lambda p, hh, ww, name=name, seed=seed: make_image(name, hh, ww, seed), "chain")


def run(compare, make_source, formats, cases_fn, failures, stats):
    """compare(label, source, **KW) -> (cpu_frame, gpu_frame); formats: iterable of (name, vs format id); cases_fn() returns a fresh case iterator."""
    for name, fmt in formats:
        for label, w, h, painter, kind in cases_fn():
            source = make_source(fmt, w, h, painter)
            cpu, gpu = compare(f"{label}-{name}", source, **KW)
            nonzero = int(sum(np.count_nonzero(np.asarray(cpu[p])) for p in range(cpu.format.num_planes)))
            stats["cases"] += 1
            if kind == "negative" and "noseed" in label and nonzero != 0:
                failures.append((label, name, "no strong pixel but the CPU output has edges", nonzero))
            if kind == "chain":
                stats["chain_cases"] += 1
                stats["chain_nonempty"] += nonzero > 0
