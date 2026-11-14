"""Memory: VRAM (nvidia-smi total delta + per-process) and host RSS for a graph at N threads.
usage: mem.py plugin backend fmt content modes sigma res threads frames"""
import sys, time, json, os, subprocess, threading
import tcprof
from tcprof import core
plugin, be, fmt, content, modes, sigma, res, th, nfr = sys.argv[1:10]
w, h = (1920, 1080) if res == "1080" else (3840, 2160)
th = int(th); nfr = int(nfr)
def status():
    d = {}
    for l in open("/proc/self/status"):
        if l.startswith(("VmRSS", "VmHWM")): k, v = l.split(":"); d[k] = int(v.split()[0]) / 1024.0
    return d
def smi_total():
    return float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().strip())
def smi_proc():
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).decode()
    for l in out.splitlines():
        p = [x.strip() for x in l.split(",")]
        if p and p[0] == str(os.getpid()): return float(p[1])
    return None
tcprof.load_plugin(plugin)
core.num_threads = th; core.max_cache_size = 512
spec = dict(backend=be, w=w, h=h, fmt=fmt, content=content, modes=[int(m) for m in modes.split(",") if m != ""], sigma=float(sigma), op=1, opt=0,
            shape="tc" if modes else "updown", threads=th)
tcprof.make_source(content, w, h, fmt, 4)   # cache source frames first (their memory is baseline)
time.sleep(0.5)
base_total = min(smi_total() for _ in range(3)); rss0 = status()
samples = []; procs = []; stop = False
def sampler():
    while not stop:
        try: samples.append(smi_total()); procs.append(smi_proc())
        except Exception: pass
        time.sleep(0.1)
t = threading.Thread(target=sampler); t.start()
clip = tcprof.build(spec, 20 + nfr)
tcprof.fetch(clip[:20], th)
t0 = time.perf_counter(); n = tcprof.fetch(clip[20:20 + nfr], th); wall = time.perf_counter() - t0
rss1 = status()
stop = True; t.join()
pp = [p for p in procs if p is not None]
print(json.dumps(dict(be=be, fmt=fmt, content=content, modes=modes, res=res, threads=th, frames=n, fps=n / wall,
                      vram_base_mib=base_total, vram_peak_total_mib=max(samples) if samples else None,
                      vram_delta_peak_mib=(max(samples) - base_total) if samples else None,
                      vram_proc_peak_mib=max(pp) if pp else None,
                      rss_before_graph_mib=rss0["VmRSS"], rss_after_mib=rss1["VmRSS"], rss_hwm_mib=rss1["VmHWM"])), flush=True)
