import sys, json, argparse, os, time
import tcprof
from suites import SUITES

ap = argparse.ArgumentParser()
ap.add_argument("suite"); ap.add_argument("out")
ap.add_argument("--plugin", default=os.environ.get("PLUGIN"))
ap.add_argument("--reps", type=int, default=5); ap.add_argument("--target", type=float, default=2.0)
ap.add_argument("--plugin-force", action="store_true")
ap.add_argument("--smi", default=None)
ap.add_argument("--part", default=None, help="i/n: run only slice i of n")
a = ap.parse_args()
specs = SUITES[a.suite]()
if a.part:
    i, n = map(int, a.part.split("/")); specs = specs[i::n]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.dumps(json.loads(l)["spec"], sort_keys=True))
        except Exception: pass
if any(s["backend"] != "upstream" for s in specs) or a.plugin_force:
    tcprof.load_plugin(a.plugin)
smi = None
if a.smi:
    import subprocess
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,clocks.gr,temperature.gpu,power.draw", "--format=csv,noheader,nounits", "-lms", "200"],
                           stdout=open(a.smi, "a"))
t00 = time.time()
with open(a.out, "a") as fo:
    for i, s in enumerate(specs):
        k = json.dumps(s, sort_keys=True)
        if k in done: continue
        try:
            r = tcprof.run_case(s, reps=a.reps, target_s=a.target)
        except Exception as e:
            r = dict(spec=s, error=repr(e))
        fo.write(json.dumps(r) + "\n"); fo.flush()
        print(f"[{i+1}/{len(specs)} {time.time()-t00:.0f}s]", s.get("backend"), s.get("shape"), s.get("fmt"), s.get("modes"), s.get("sigma"),
              s.get("content"), "fps=%.1f" % r.get("fps_median", -1), r.get("error", ""), flush=True)

if smi: smi.terminate()
