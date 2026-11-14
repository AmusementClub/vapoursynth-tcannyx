import glob, json, os, re
import sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/instr"
rows = []
for out in sorted(glob.glob(D + "/*.out")):
    tag = os.path.basename(out)[:-4]
    try: j = json.loads(open(out).read().strip().splitlines()[-1])
    except Exception: j = {}
    err = open(out[:-4] + ".err").read()
    rows.append((tag, j, err))
def perframe(err, name):
    m = re.search(re.escape(name) + r".*?per-frame=\s*([\d.]+) ms", err)
    return float(m.group(1)) if m else None
def calls(err, name):
    m = re.search(re.escape(name) + r".*?calls=(\d+)", err); return int(m.group(1)) if m else None
def hyst(err):
    m = re.search(r"hysteresis frames=(\d+) batches/frame mean=([\d.]+) min=(\d+) max=(\d+) \| passes-executed/frame mean=([\d.]+) min=(\d+) max=(\d+)", err)
    return m.groups() if m else None
print("## hysteresis (tag: fps | frames, batches mean/min/max, passes mean/min/max)")
for tag, j, err in rows:
    if tag.startswith(("vk_", "cu_", "depth_")):
        h = hyst(err)
        if h: print(f"{tag}: fps={j.get('fps',0):.0f} | frames={h[0]} batches {h[1]}/{h[2]}/{h[3]} passes {h[4]}/{h[5]}/{h[6]}")
print("## vulkan per-frame thread-time ms")
for tag, j, err in rows:
    if tag.startswith("vk_") and "TCANNY_PROF vulkan" in err:
        n = int(re.search(r"frames=(\d+)", err).group(1))
        print(tag, "fps=%.0f" % j["fps"], "total=%s" % perframe(err, "total tcannyProcessVulkan"), "newFrame=%s" % perframe(err, "newGPUVideoFrame"), "plan=%s" % perframe(err, "plan()"),
              "acquire=%s" % perframe(err, "gpuExecAcquire"), "record=%s" % perframe(err, "record()"), "submit=%s" % perframe(err, "gpuExecSubmit"),
              "wait=%s" % perframe(err, "gpuExecWaitValue"), "subs/frame=%.2f" % ((calls(err, "gpuExecSubmit") or 0) / n), "waits/frame=%.2f" % ((calls(err, "gpuExecWaitValue") or 0) / n))
print("## cuda per-frame thread-time ms")
keys = ["total tcannyProcessCuda", "waitGPUFrame", "selectDevice", "newGPUVideoFrame", "export+import", "cudaStreamCreate", "weights cudaMalloc", "cudaMallocPitch", "launch blur", "launch tcanny_kernel",
        "hysteresis loop total", "cudaMalloc(changed)", "memset+16", "memcpyAsync flag", "stream sync per batch", "finalize launch", "final stream sync", "dtor: stream", "dtor: cudaFree", "dtor: destroyExternalMemory",
        "publishCudaSignal", "createGPUTimeline", "exportGPUSemaphore", "cudaImportExternalSemaphore", "cudaSignalExternal", "cudaStreamSynchronize(signal)", "dtor: semaphore"]
for tag, j, err in rows:
    if tag.startswith("cu_") and "TCANNY_PROF cuda" in err:
        print(tag, "fps=%.0f" % j["fps"], " ".join(f"{k.split()[0]}={perframe(err,k)}" for k in keys))
print("## cpu stage shares")
for tag, j, err in rows:
    if tag.startswith("cpu_"):
        sh = re.findall(r"^\s+(.+?)\s{2,}total=\s*[\d.]+ ms per-frame=\s*([\d.]+) ms\s+share=\s*([\d.]+)%", err, re.M)
        print(tag, "fps=%.0f" % j.get("fps", 0), "cores=%.1f" % j.get("cpu_cores", 0), "; ".join(f"{n[:12]}={pf}ms/{s}%" for n, pf, s in sh))
