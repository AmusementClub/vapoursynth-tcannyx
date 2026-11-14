"""Aggregate results/*.jsonl into markdown tables (results/tables_*.md)."""
import json, sys, os, collections
R = "results"
def load(name):
    p = f"{R}/{name}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def bk(s):
    b = s["backend"]
    if b == "cpu": return f"cpu opt{s['opt']}"
    if b == "upstream": return f"upstream opt{s['opt']}"
    return b
def cell(r):
    if "error" in r: return "ERR"
    return f"{r['fps_median']:.1f} [{r['fps_min']:.1f}-{r['fps_max']:.1f}] / {r['ms_median']:.2f}"
def spread(r):
    return (r['fps_max'] - r['fps_min']) / r['fps_median'] if 'fps_median' in r else 0
def table(rows, rowf, colf, rowhdr, colorder=None, f=cell):
    d = collections.OrderedDict(); cols = []
    for r in rows:
        s = r["spec"]; k = rowf(s); c = colf(s)
        d.setdefault(k, {})[c] = r
        if c not in cols: cols.append(c)
    if colorder: cols = [c for c in colorder if c in cols] + [c for c in cols if c not in colorder]
    out = ["| " + rowhdr + " | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for k, v in d.items():
        out.append("| " + str(k) + " | " + " | ".join(f(v[c]) if c in v else "-" for c in cols) + " |")
    return "\n".join(out)
BE = ["cpu opt1", "cpu opt0", "vulkan", "cuda"]
def main():
    out = {}
    base = load("baselines")
    out["baselines"] = table(base, lambda s: f"{s['w']}x{s['h']} {s['fmt']} {s['content']}", lambda s: s["shape"] + ("" if s["shape"] == "src" else "(" + s["backend"] + ")"), "case (fps [min-max] / ms median)")
    mm = load("matrix_main")
    for fmt in ("GRAYS", "GRAY8"):
        rows = [r for r in mm if r["spec"]["fmt"] == fmt]
        out["matrix_main_" + fmt] = table(rows, lambda s: f"mode {s['modes'][0]} sigma {s['sigma']} {s['content']}", lambda s: bk(s), "1080p " + fmt + " op1: fps [min-max] / ms", BE)
    of = load("matrix_ofat")
    out["ofat"] = table(of, lambda s: f"{s['w']}x{s['h']} {s['fmt']} op{s['op']} sigma{s['sigma']} mode {s['modes'][0]} {s['content']}", bk, "case", BE)
    rs = load("resident")
    out["resident"] = table(rs, lambda s: f"{s['w']}x{s['h']} {s['fmt']} modes={s['modes']}", lambda s: s["backend"], "chain", ["vulkan", "cuda"])
    sc = load("scaling")
    for mode in (0, 1):
        out[f"scaling_mode{mode}"] = table([r for r in sc if r["spec"]["modes"] == [mode]], lambda s: s["threads"], bk, "threads (fps [min-max] / ms)", BE)
    for k, v in out.items():
        open(f"{R}/tables_{k}.md", "w").write(v + "\n")
    return out
if __name__ == "__main__":
    o = main()
    for k, v in o.items():
        print("##", k); print(v); print()
