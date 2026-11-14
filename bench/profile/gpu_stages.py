"""Per-stage GPU time per frame from an nsys sqlite export, for either backend, in one table.
Vulkan: needs a -Dprofiling=true build (stage debug labels) traced with --trace=vulkan,vulkan-annotations
--vulkan-gpu-workload=true. CUDA: kernel names (CUPTI kernel activity), no profiling build needed.
usage: gpu_stages.py <file.sqlite> [frames]   (frames defaults to the number of final stage executions; planes count once each)"""
import collections, sqlite3, statistics, sys

CUDA_STAGE = {"tcanny_blur_v": "blur_v", "tcanny_blur_h": "blur_h", "tcanny_kernel": "gradient/store", "tcanny_nms": "nms",
              "tcanny_pack": "pack", "tcanny_hysteresis": "hysteresis", "tcanny_finalize": "finalize"}
ORDER = ["blur_v", "blur_h", "gradient", "gradient/store", "store", "nms", "pack", "hysteresis", "finalize"]


def vulkan(c):
    ids = dict(c.execute("select id,value from StringIds"))
    rows = collections.defaultdict(list)
    for s, e, n, t in c.execute("select start,end,nameId,textId from VULKAN_WORKLOAD"):
        label = ids.get(t)
        if ids.get(n) == "vkCmdBeginDebugUtilsLabelEXT" and label:
            rows[label].append((e - s) / 1e3)
    return rows


def cuda(c):
    ids = dict(c.execute("select id,value from StringIds"))
    rows = collections.defaultdict(list)
    for s, e, n in c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        name = ids.get(n, "")
        for key, stage in CUDA_STAGE.items():
            if name.startswith(key):
                rows[stage].append((e - s) / 1e3)
    return rows


def main():
    c = sqlite3.connect(sys.argv[1])
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    rows = {}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        rows = cuda(c)
        backend = "cuda"
    if not rows and "VULKAN_WORKLOAD" in tables:
        rows = vulkan(c)
        backend = "vulkan"
    if not rows:
        sys.exit("no labelled stages found (Vulkan needs a -Dprofiling=true build and --trace=vulkan,vulkan-annotations)")
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else len(rows.get("finalize") or rows.get("store") or rows.get("gradient/store") or rows.get("gradient"))
    print(f"{sys.argv[1].split('/')[-1]}  backend={backend}  frames={frames}")
    print(f"  {'stage':<15}{'n/frame':>9}{'median us':>11}{'us/frame':>10}{'share':>8}")
    total = sum(sum(v) for v in rows.values()) / frames
    for stage in ORDER + sorted(set(rows) - set(ORDER)):
        v = rows.get(stage)
        if not v:
            continue
        per = sum(v) / frames
        print(f"  {stage:<15}{len(v) / frames:>9.2f}{statistics.median(v):>11.1f}{per:>10.1f}{100 * per / total:>7.1f}%")
    print(f"  {'total':<15}{'':>9}{'':>11}{total:>10.1f}")


main()
