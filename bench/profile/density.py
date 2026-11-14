import sys, json, time, numpy as np
import tcprof
from tcprof import core
tcprof.load_plugin(sys.argv[1])
core.num_threads = 8
out = []
for (w, h) in ((1920, 1080), (3840, 2160)):
    for fmt in ("GRAYS", "GRAY8", "YUV420P8"):
        for content in ("natural", "busy", "flat"):
            src = tcprof.make_source(content, w, h, fmt, 4)
            cpu = core.tcannyx.TCanny(src, sigma=1.5, mode=0, op=1, opt=1, backend="cpu")
            g1 = core.tcannyx.TCanny(src, sigma=1.5, mode=1, op=1, opt=1, backend="cpu")
            row = dict(w=w, h=h, fmt=fmt, content=content, edge_frac=[], mode1_mean=[], strong_frac=[])
            # strong-only (t_l raised to just below t_h) isolates seeds
            for n in range(4):
                f = cpu.get_frame(n); a = np.asarray(f[0])
                row["edge_frac"].append(float(np.count_nonzero(a)) / a.size)
                m = np.asarray(g1.get_frame(n)[0]).astype(np.float64)
                scale = 1.0 if fmt == "GRAYS" else 255.0
                row["mode1_mean"].append(float(m.mean() / scale))
            if fmt == "GRAYS" and w == 1920:  # GPU parity on this content, frame 0
                ref = np.asarray(cpu.get_frame(0)[0]).copy()
                for be in ("vulkan", "cuda"):
                    gp = core.tcannyx.TCanny(src.std.GPUUpload(), sigma=1.5, mode=0, op=1, backend=be).std.GPUDownload()
                    a = np.asarray(gp.get_frame(0)[0])
                    row[f"{be}_mismatch_px_frame0"] = int(np.count_nonzero(a != ref))
            out.append(row)
            print(row, flush=True)
json.dump(out, open("results/density.json", "w"), indent=1)
