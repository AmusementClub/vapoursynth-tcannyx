"""Shared harness: deterministic cached synthetic sources, graph builder, timing."""
import os, sys, time, json, resource, statistics, hashlib
import numpy as np
import vapoursynth as vs

core = vs.core
CACHE = os.environ.get("TCPROF_CACHE", os.path.expanduser("~/.cache/tcprof"))
os.makedirs(CACHE, exist_ok=True)
NDISTINCT = 4
FORMATS = {n: getattr(vs, n) for n in ("GRAY8", "GRAY16", "GRAYS", "YUV420P8", "YUV444PS")}

# ---------------------------------------------------------------- content
def _smooth(rng, h, w, cell):
    gh, gw = h // cell + 2, w // cell + 2
    g = rng.random((gh, gw), dtype=np.float32)
    ys = np.arange(h, dtype=np.float32) / cell; y0 = ys.astype(np.int32); fy = (ys - y0)[:, None]
    xs = np.arange(w, dtype=np.float32) / cell; x0 = xs.astype(np.int32); fx = (xs - x0)[None, :]
    rows = g[y0] * (1 - fy) + g[y0 + 1] * fy
    return rows[:, x0] * (1 - fx) + rows[:, x0 + 1] * fx

def gen_natural(rng, h, w):
    s = max(h / 1080.0, 0.25)
    img = np.zeros((h, w), np.float32)
    for cell, amp in ((256, .35), (128, .2), (64, .12), (32, .08), (16, .05), (8, .03)):
        img += amp * _smooth(rng, h, w, max(2, int(cell * s)))
    img = 0.15 + 0.6 * (img - img.min()) / (img.max() - img.min())
    for _ in range(30):  # hard-edged rectangles
        hh, ww = int(rng.integers(40, 400) * s), int(rng.integers(40, 500) * s)
        y, x = int(rng.integers(0, max(1, h - hh))), int(rng.integers(0, max(1, w - ww)))
        img[y:y + hh, x:x + ww] += rng.uniform(-.25, .25)
    yy, xx = np.ogrid[:h, :w]
    for _ in range(20):  # ellipses
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(20, 200) * s, rng.integers(20, 250) * s
        m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        img[m] += rng.uniform(-.25, .25)
    for _ in range(40):  # thin lines
        y0, x0, y1, x1 = rng.integers(0, h), rng.integers(0, w), rng.integers(0, h), rng.integers(0, w)
        n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
        ys = np.linspace(y0, y1, n).astype(int); xs = np.linspace(x0, x1, n).astype(int)
        v = rng.uniform(-.3, .3)
        for dy in (0, 1):
            for dx in (0, 1):
                img[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)] += v
    for _ in range(10):  # text-like blocks: random 4x5 glyph cells scaled up
        bh, bw = int(rng.integers(60, 140) * s), int(rng.integers(300, 600) * s)
        y, x = int(rng.integers(0, max(1, h - bh))), int(rng.integers(0, max(1, w - bw)))
        k = max(1, int(round(2 * s)))
        cells = (rng.random((bh // k + 1, bw // k + 1)) < 0.4).astype(np.float32)
        cells = cells * np.tile(np.array([1, 1, 1, 0], np.float32), (cells.shape[0], cells.shape[1] // 4 + 1))[:, :cells.shape[1]]
        m = np.kron(cells, np.ones((k, k), np.float32))[:bh, :bw]
        img[y:y + bh, x:x + bw] = img[y:y + bh, x:x + bw] * (1 - m) + m * rng.uniform(0.05, 0.15)
    img += rng.normal(0, 0.006, (h, w)).astype(np.float32)  # mild grain
    return np.clip(img, 0, 1)

def gen_busy(rng, h, w):
    yy, xx = np.ogrid[:h, :w]
    tex = 0.12 * np.sin(xx * 1.3) * np.sin(yy * 1.7)
    img = 0.5 + 0.4 * (rng.random((h, w), dtype=np.float32) - 0.5) * 2 + tex
    return np.clip(img, 0, 1).astype(np.float32)

def gen_flat(rng, h, w):
    return np.full((h, w), 0.5, np.float32)

GENS = {"natural": gen_natural, "busy": gen_busy, "flat": gen_flat}

def planes_float(content, w, h, fmt, idx):
    """list of float32 arrays per plane, value range [0,1] (chroma of float YUV centred at 0 later)."""
    f = core.get_video_format(FORMATS[fmt])
    out = []
    for p in range(f.num_planes):
        pw = w >> (f.subsampling_w if p else 0); ph = h >> (f.subsampling_h if p else 0)
        seed = int(hashlib.md5(f"{content}/{w}x{h}/{idx}/{p}".encode()).hexdigest()[:8], 16)
        a = GENS[content](np.random.default_rng(seed), ph, pw)
        if p:  # chroma: weaker amplitude around mid grey
            a = 0.5 + 0.3 * (a - 0.5)
        out.append(a)
    return out

def to_native(arrs, fmt):
    f = core.get_video_format(FORMATS[fmt])
    res = []
    for p, a in enumerate(arrs):
        if f.sample_type == vs.INTEGER:
            peak = (1 << f.bits_per_sample) - 1
            res.append(np.rint(a * peak).astype(np.uint8 if f.bytes_per_sample == 1 else np.uint16))
        else:
            res.append((a - 0.5 if (p and f.color_family == vs.YUV) else a).astype(np.float32))
    return res

def get_native(content, w, h, fmt, idx):
    fn = f"{CACHE}/{content}_{w}x{h}_{fmt}_{idx}.npz"
    if os.path.exists(fn):
        z = np.load(fn)
        return [z[f"p{i}"] for i in range(len(z.files))]
    nat = to_native(planes_float(content, w, h, fmt, idx), fmt)
    np.savez(fn + ".tmp.npz", **{f"p{i}": a for i, a in enumerate(nat)})
    os.replace(fn + ".tmp.npz", fn)
    return nat

_SRC = {}
def make_source(content, w, h, fmt, length):
    """CPU clip whose frames are cached, pre-filled VideoFrames (cycled), so generation cost is ~0."""
    key = (content, w, h, fmt)
    if key not in _SRC:
        blank = core.std.BlankClip(width=w, height=h, format=FORMATS[fmt], length=1)
        n = 1 if content == "flat" else NDISTINCT
        frames = []
        for i in range(n):
            fr = blank.get_frame(0).copy()
            for p, a in enumerate(get_native(content, w, h, fmt, i)):
                np.asarray(fr[p])[:] = a
            frames.append(fr)
        _SRC[key] = frames
    frames = _SRC[key]
    blank = core.std.BlankClip(width=w, height=h, format=FORMATS[fmt], length=length)
    return core.std.ModifyFrame(blank, blank, lambda n, f: frames[n % len(frames)])

# ---------------------------------------------------------------- graphs
def build(spec, length):
    """spec keys: backend(cpu|vulkan|cuda|upstream|none) shape(src|updown|tc) content w h fmt modes[list] sigma op opt"""
    clip = make_source(spec["content"], spec["w"], spec["h"], spec["fmt"], length)
    shape = spec.get("shape", "tc")
    be = spec["backend"]
    if shape == "src":
        return clip
    gpu = be in ("vulkan", "cuda")
    if gpu:
        clip = clip.std.GPUUpload()
    if shape != "updown":
        for m in spec["modes"]:
            kw = dict(sigma=spec.get("sigma", 1.5), mode=m, op=spec.get("op", 1))
            if be in ("cpu", "upstream"):
                kw["opt"] = spec.get("opt", 0)
            if be != "upstream":
                kw["backend"] = be
            if "t_h" in spec: kw["t_h"] = spec["t_h"]; kw["t_l"] = spec["t_l"]
            clip = (core.tcanny.TCanny if be == "upstream" else core.tcannyx.TCanny)(clip, **kw)  # upstream r12/r14 keeps its own namespace
    if gpu:
        clip = clip.std.GPUDownload()
    return clip

def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime

def fetch(clip, threads):
    n = 0
    for _ in clip.frames(prefetch=threads, backlog=threads * 2):
        n += 1
    return n

WARM = 6
def run_case(spec, reps=5, target_s=2.0, min_n=6, max_n=2000):
    threads = spec["threads"]
    core.num_threads = threads
    core.max_cache_size = 512
    # probe to size N
    clip = build(spec, 64)
    fetch(clip[:WARM], threads)
    t0 = time.perf_counter(); n = fetch(clip[WARM:WARM + 12], threads); dt = time.perf_counter() - t0
    del clip
    est = n / dt
    N = int(max(min_n, min(max_n, est * target_s)))
    runs = []
    for r in range(reps):
        clip = build(spec, WARM + N)
        fetch(clip[:WARM], threads)  # per-graph warm-up (untimed)
        tail = clip[WARM:WARM + N]
        c0 = cpu_s(); t0 = time.perf_counter()
        n = fetch(tail, threads)
        wall = time.perf_counter() - t0; cpu = cpu_s() - c0
        runs.append({"n": n, "wall": wall, "cpu": cpu, "fps": n / wall, "t_start": t0, "t_end": t0 + wall,
                     "wall_epoch_end": time.time()})
        del clip, tail
    fps = [x["fps"] for x in runs]
    med = statistics.median(fps)
    res = dict(spec=spec, N=N, reps=reps, fps_median=med, fps_min=min(fps), fps_max=max(fps),
               ms_median=1000 / med, ms_min=1000 / max(fps), ms_max=1000 / min(fps),
               cpu_cores_median=statistics.median([x["cpu"] / x["wall"] for x in runs]),
               runs=[{k: v for k, v in x.items() if k in ("n", "wall", "cpu", "fps", "wall_epoch_end")} for x in runs],
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    return res

def load_plugin(path):
    core.std.LoadPlugin(path)
