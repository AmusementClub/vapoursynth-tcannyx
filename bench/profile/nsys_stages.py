"""Median GPU duration of the first dispatches of a Vulkan mode-0 submission (nsys sqlite export).
Order after B1+D: blur V, blur H, gradient, NMS, pack, then hysteresis passes. usage: nsys_stages.py <file.sqlite> [n]"""
import collections, sqlite3, statistics, sys
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
c = sqlite3.connect(sys.argv[1])
groups = collections.OrderedDict()
for s, e, cid in c.execute("select start,end,correlationId from VULKAN_WORKLOAD order by start"):
    groups.setdefault(cid, []).append((s, e))
first = [g for g in groups.values() if len(g) >= 20]
names = ["blurV", "blurH", "gradient", "nms", "pack"] + ["pass%d" % i for i in range(n)]
print(sys.argv[1].split("/")[-1], "submissions with >=20 dispatches:", len(first))
for i in range(min(n, len(first[0]))):
    print("  %-9s %.1f us" % (names[i], statistics.median((g[i][1] - g[i][0]) / 1e3 for g in first)))
