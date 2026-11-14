"""F1: GPUUpload -> GPUDownload only, GRAYS, at frame sizes of 4, 8, 16, 32 MB; 5 repetitions each, with GPU clocks.
usage: f1_updown.py <libtcannyx.so> (the plugin is not used by the graph; it only keeps the harness identical)"""
import json, statistics, subprocess, sys, time
import numpy as np
import vapoursynth as vs
core = vs.core
core.num_threads = 4
core.max_cache_size = 512
def smi():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return [int(v) for v in out.strip().split(",")]
res = []
for side in (1024, 1448, 2048, 2896):
    mb = side * side * 4 / 2 ** 20
    blank = core.std.BlankClip(width=side, height=side, format=vs.GRAYS, length=300, color=[0.5])
    clip = blank.std.GPUUpload().std.GPUDownload()
    for f in clip[:20].frames(prefetch=4): pass
    runs = []
    for r in range(5):
        t0 = time.perf_counter(); n = 0
        for f in clip[20:220].frames(prefetch=4): n += 1
        runs.append(n / (time.perf_counter() - t0))
    clocks = smi()
    res.append(dict(frame_mb=round(mb, 1), fps_median=round(statistics.median(runs), 1), fps_min=round(min(runs), 1), fps_max=round(max(runs), 1),
                    gbps_median=round(statistics.median(runs) * mb / 1024 * 2, 1), sm_clock_mhz=clocks[0], util=clocks[1]))
    print(json.dumps(res[-1]), flush=True)
