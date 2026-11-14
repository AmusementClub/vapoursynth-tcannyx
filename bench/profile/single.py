"""One graph, one process: warm-up then timed window; used for instrumented counters / nsys / perf runs.
usage: single.py plugin backend fmt content modes(comma) sigma op res(1080|4k) threads frames [opt]"""
import sys, time, gc, json
import tcprof
from tcprof import core
plugin, be, fmt, content, modes, sigma, op, res, th, nfr = sys.argv[1:11]
opt = int(sys.argv[11]) if len(sys.argv) > 11 else 0
w, h = (1920, 1080) if res == "1080" else (3840, 2160)
th = int(th); nfr = int(nfr)
tcprof.load_plugin(plugin)
core.num_threads = th; core.max_cache_size = 512
spec = dict(backend=be, w=w, h=h, fmt=fmt, content=content, modes=[int(m) for m in modes.split(",") if m != ""], sigma=float(sigma), op=int(op), opt=opt,
            shape="tc" if modes else "updown", threads=th)
W = 20
clip = tcprof.build(spec, W + nfr)
tcprof.fetch(clip[:W], th)
tail = clip[W:W + nfr]
c0 = tcprof.cpu_s(); t0 = time.perf_counter()
n = tcprof.fetch(tail, th)
wall = time.perf_counter() - t0; cpu = tcprof.cpu_s() - c0
print(json.dumps(dict(be=be, fmt=fmt, content=content, modes=modes, sigma=sigma, res=res, threads=th, frames=n, wall=wall, fps=n / wall, cpu_cores=cpu / wall)), flush=True)
del clip, tail
gc.collect()
