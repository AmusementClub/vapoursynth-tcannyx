"""GPU-resident TCanny throughput: the source is uploaded once (4 distinct frames, looped on the GPU) and no frame is
downloaded except the last one, which only waits for completion. A chain of K filters gives the per-filter cost as a
slope; transfers cancel out of (us at K=4 - us at K=1) / 3.
usage: resident.py <plugin> <backend> <fmt> <mode> <K> <threads> <frames> [res]   (prints one JSON line)"""
import json
import sys
import time

plugin, be, fmt, mode, K, th, nfr = sys.argv[1:8]
res = sys.argv[8] if len(sys.argv) > 8 else "1080"
import tcprof  # noqa: E402  (bench/profile, next to this script)
from tcprof import core  # noqa: E402

K, th, nfr, mode = int(K), int(th), int(nfr), int(mode)
w, h = (1920, 1080) if res == "1080" else (3840, 2160)
tcprof.load_plugin(plugin)
core.num_threads = th
core.max_cache_size = 2048
src = tcprof.make_source("natural", w, h, fmt, 4).std.GPUUpload()
warm = 24
clip = src.std.Loop(times=(warm + nfr) // 4 + 2)
for _ in range(K):
    clip = core.tcannyx.TCanny(clip, sigma=1.5, mode=mode, op=1, backend=be)
for _ in clip[:warm].frames(prefetch=th, backlog=th * 2):
    pass
t0 = time.perf_counter()
n = 0
for _ in clip[warm:warm + nfr].frames(prefetch=th, backlog=th * 2):
    n += 1
clip[warm + nfr - 1].std.GPUDownload().get_frame(0)  # completion of the last frame (in-order GPU work)
wall = time.perf_counter() - t0
print(json.dumps(dict(be=be, fmt=fmt, mode=mode, K=K, threads=th, res=res, frames=n, fps=n / wall,
                      us_per_frame=wall / n * 1e6)), flush=True)
