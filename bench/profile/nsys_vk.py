"""Analyse Vulkan GPU workloads from an nsys sqlite export (per-dispatch durations grouped by submission)."""
import sqlite3, sys, json, statistics, collections
def analyse(fn, skip_frames_groups=0):
    c = sqlite3.connect(fn)
    rows = c.execute("select start,end,correlationId from VULKAN_WORKLOAD order by start").fetchall()
    groups = collections.OrderedDict()
    for s, e, cid in rows: groups.setdefault(cid, []).append((s, e))
    gl = list(groups.values())
    sizes = collections.Counter(len(g) for g in gl)
    t0, t1 = rows[0][0], rows[-1][1]
    busy = 0; cur_s, cur_e = rows[0][0], rows[0][1]
    for s, e, _ in rows[1:]:
        if s > cur_e: busy += cur_e - cur_s; cur_s, cur_e = s, e
        else: cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    out = dict(file=fn, workloads=len(rows), groups=len(gl), group_sizes=dict(sizes.most_common(12)), span_ms=(t1 - t0) / 1e6, union_busy_ms=busy / 1e6)
    # per-position durations in groups of the 'first' kind (3 main-stage dispatches then hysteresis)
    first = [g for g in gl if len(g) >= 3 and len(g) == max(sizes, key=lambda k: (sizes[k] if k >= 3 else 0))]
    if first:
        for pos, nm in enumerate(["stage1 blurV?", "stage2 blurH?", "stage3 grad/NMS?"]):
            d = [(g[pos][1] - g[pos][0]) / 1e3 for g in first if len(g) > pos]
            out[f"first_group_pos{pos}_{nm}_us_median"] = statistics.median(d)
        out["first_group_size"] = len(first[0])
        hyst = [(e - s) / 1e3 for g in first for (s, e) in g[3:]]
        if hyst: out["first_group_hyst_pass_us_median"] = statistics.median(hyst)
    singles = [g for g in gl if len(g) == 1]
    if singles: out["size1_group_us_median(finalize or mode1 single)"] = statistics.median((g[0][1] - g[0][0]) / 1e3 for g in singles)
    other = [g for g in gl if len(g) > 1 and (not first or len(g) != len(first[0]))]
    if other:
        per = [(g[-1][1] - g[0][0]) / len(g) / 1e3 for g in other]
        out["hyst_only_groups_us_per_pass_median"] = statistics.median(per)
        out["hyst_only_groups_n"] = len(other)
    # inter-group gap (GPU idle between consecutive submissions), only when groups don't overlap
    gaps = []
    for a, b in zip(gl, gl[1:]):
        gap = b[0][0] - a[-1][1]
        if 0 <= gap < 5e6: gaps.append(gap / 1e3)
    if gaps: out["inter_group_gap_us_median"] = statistics.median(gaps); out["inter_group_gap_us_mean"] = statistics.mean(gaps); out["gaps_n"] = len(gaps)
    return out
if __name__ == "__main__":
    res = {fn: analyse(fn) for fn in sys.argv[1:]}
    print(json.dumps(res, indent=1))
