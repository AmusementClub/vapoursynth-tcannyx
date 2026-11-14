"""Acceptance-set table. usage: accept_report.py <label> [<label2> ...]  (reads results/accept_<label>.jsonl)
With several labels the cells show median fps for each label side by side."""
import json, sys

def load(label):
    out = {}
    for l in open(f"results/accept_{label}.jsonl"):
        r = json.loads(l)
        if "error" in r:
            continue
        s = r["spec"]
        key = (s["backend"], s.get("opt", ""), s["fmt"], s["shape"], tuple(s["modes"]), s.get("sigma"), s["threads"])
        out[key] = r
    return out

labels = sys.argv[1:]
data = [load(l) for l in labels]
keys = list(data[0].keys())
print("| backend | fmt | shape | modes | sigma | threads | " + " | ".join(f"{l} fps [min-max]" for l in labels) + " |")
print("|" + "---|" * (6 + len(labels)))
for k in keys:
    be, opt, fmt, shape, modes, sigma, th = k
    name = be + (f" opt{opt}" if be == "cpu" else "")
    cells = []
    for d in data:
        r = d.get(k)
        cells.append(f"{r['fps_median']:.1f} [{r['fps_min']:.1f}-{r['fps_max']:.1f}]" if r else "-")
    print(f"| {name} | {fmt} | {shape} | {list(modes)} | {sigma} | {th} | " + " | ".join(cells) + " |")
