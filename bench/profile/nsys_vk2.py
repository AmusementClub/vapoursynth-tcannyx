import sqlite3, sys, json, statistics, collections
res = {}
for fn in sys.argv[1:]:
    c = sqlite3.connect(fn)
    rows = c.execute("select start,end,correlationId from VULKAN_WORKLOAD order by start").fetchall()
    g = collections.OrderedDict()
    for s, e, cid in rows: g.setdefault(cid, []).append((s, e))
    gl = [(min(s for s, e in l), max(e for s, e in l), len(l)) for l in g.values()]
    gl.sort()
    bysize = collections.defaultdict(list)
    for s, e, n in gl: bysize[n].append((e - s) / 1e3)
    gaps = [(b[0] - a[1]) / 1e3 for a, b in zip(gl, gl[1:]) if 0 <= b[0] - a[1] < 10e6]
    res[fn.split("/")[-1][:-7]] = dict(
        groups=len(gl),
        span_by_group_size_us={n: dict(count=len(v), median=round(statistics.median(v), 1), us_per_dispatch_minus2=round(statistics.median(v) / max(1, n - 2), 2)) for n, v in sorted(bysize.items())},
        total_span_ms=round(sum(e - s for s, e, n in gl) / 1e6, 2), total_gap_ms=round(sum(gaps) / 1e3, 2), median_gap_us=round(statistics.median(gaps), 1) if gaps else None,
        wall_ms=round((gl[-1][1] - gl[0][0]) / 1e6, 1))
print(json.dumps(res, indent=1))
