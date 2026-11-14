import os
T = int(os.environ.get("TC_THREADS", "16"))
R1080 = (1920, 1080); R4K = (3840, 2160)

def S(backend, res, fmt, content, modes, sigma=1.5, op=1, opt=0, shape="tc", threads=None):
    d = dict(backend=backend, w=res[0], h=res[1], fmt=fmt, content=content, modes=list(modes), sigma=sigma, op=op,
             shape=shape, threads=threads or T)
    if backend in ("cpu", "upstream"): d["opt"] = opt
    return d

# (label, backend, opt)
CPU = [("cpu", 1), ("cpu", 0)]
def backends_for(fmt, gpu_ok_cuda=None):
    out = [("cpu", 1), ("cpu", 0), ("vulkan", 0)]
    if fmt in ("GRAYS", "YUV444PS"): out.append(("cuda", 0))
    return out

def mk(b, res, fmt, content, mode, **kw):
    be, opt = b
    return S(be, res, fmt, content, [mode], opt=opt, **kw)

def baselines():
    out = []
    for res, fmts in ((R1080, ["GRAY8", "GRAY16", "GRAYS", "YUV420P8", "YUV444PS"]), (R4K, ["GRAY8", "GRAY16", "GRAYS"])):
        for fmt in fmts:
            for c in ("natural", "busy", "flat") if fmt in ("GRAY8", "GRAYS") else ("natural",):
                out.append(S("cpu", res, fmt, c, [], shape="src"))
                out.append(S("vulkan", res, fmt, c, [], shape="updown"))
    out.append(S("cuda", R1080, "YUV444PS", "natural", [], shape="updown"))
    return out

def matrix_main():
    out = []
    for fmt in ("GRAYS", "GRAY8"):
        for b in backends_for(fmt):
            for mode in (-1, 0, 1):
                for sigma in (0.0, 1.5, 4.0):
                    for c in (("natural", "busy", "flat") if mode == 0 else ("natural", "busy")):
                        out.append(mk(b, R1080, fmt, c, mode, sigma=sigma))
    return out

def matrix_ofat():
    out = []
    # op variations at 1080p GRAYS/GRAY8, modes 0/1
    for fmt in ("GRAYS", "GRAY8"):
        for b in backends_for(fmt):
            for op in (2, 6):
                for mode in (0, 1):
                    for c in ("natural", "busy"):
                        out.append(mk(b, R1080, fmt, c, mode, op=op))
    # other formats at 1080p
    for fmt in ("GRAY16", "YUV420P8", "YUV444PS"):
        for b in backends_for(fmt):
            for mode in (-1, 0, 1):
                for c in ("natural", "busy"):
                    out.append(mk(b, R1080, fmt, c, mode))
    # 4K
    for fmt in ("GRAY8", "GRAY16", "GRAYS"):
        for b in backends_for(fmt):
            for mode in (-1, 0, 1):
                for c in ("natural", "busy"):
                    out.append(mk(b, R4K, fmt, c, mode))
    for fmt in ("GRAYS", "GRAY8"):
        for b in backends_for(fmt):
            for sigma in (0.0, 4.0):
                for mode in (0, 1):
                    out.append(mk(b, R4K, fmt, "natural", mode, sigma=sigma))
    return out

def resident():
    out = []
    for res, combos in ((R1080, [("vulkan", "GRAYS"), ("vulkan", "GRAY8"), ("cuda", "GRAYS")]), (R4K, [("vulkan", "GRAYS"), ("cuda", "GRAYS")])):
        for be, fmt in combos:
            chains = [[m] * k for m in (1, -1, 0) for k in (1, 2, 4)] if res == R1080 else [[1] * k for k in (1, 2, 4)] + [[0]]
            if res == R1080:
                chains += [[-1] * 3, [-1] * 3 + [0]]
            for ch in chains:
                out.append(S(be, res, fmt, "natural", ch))
    return out

def scaling():
    out = []
    for th in (1, 2, 4, 8, 16, 32):
        for b in (("cpu", 1), ("cpu", 0), ("vulkan", 0), ("cuda", 0)):
            for mode in (0, 1):
                out.append(mk(b, R1080, "GRAYS", "natural", mode, threads=th))
    return out

def upstream():
    out = []
    for res, fmts in ((R1080, ["GRAY8", "GRAY16", "GRAYS", "YUV420P8", "YUV444PS"]), (R4K, ["GRAY8", "GRAYS"])):
        for fmt in fmts:
            for mode in (-1, 0, 1):
                for opt in (1, 0):
                    for sigma in (1.5,) if (res == R4K or fmt not in ("GRAY8", "GRAYS")) else (0.0, 1.5, 4.0):
                        for c in ("natural", "busy"):
                            out.append(S("upstream", res, fmt, c, [mode], sigma=sigma, opt=opt))
    return out

def anom():
    out = []
    for th in (1, 4, 16):
        for res, fmt in ((R1080, "GRAYS"), (R1080, "YUV444PS"), (R4K, "GRAYS"), (R4K, "GRAY8")):
            out.append(S("vulkan", res, fmt, "natural", [], shape="updown", threads=th))
            out.append(S("vulkan", res, fmt, "natural", [-1], sigma=0.0, threads=th))
    return out

def ctl():
    out = []
    for fmt in ("GRAY8", "GRAYS"):
        for mode in (-1, 0, 1):
            for opt in (0, 1):
                out.append(S("cpu", R1080, fmt, "natural", [mode], opt=opt))
    return out

def upsrc():
    return [S("upstream", R1080, f, "natural", [], shape="src") for f in ("GRAY8", "GRAYS")] + [S("upstream", R4K, "GRAYS", "natural", [], shape="src")]

def smoke():
    return [S("cpu", R1080, "GRAYS", "natural", [], shape="src"),
            mk(("cpu", 0), R1080, "GRAYS", "natural", 0),
            mk(("vulkan", 0), R1080, "GRAYS", "natural", 0),
            mk(("cuda", 0), R1080, "GRAYS", "natural", 1)]

SUITES = dict(baselines=baselines, matrix_main=matrix_main, matrix_ofat=matrix_ofat, resident=resident, scaling=scaling,
              upstream=upstream, smoke=smoke, anom=anom, ctl=ctl, upsrc=upsrc)

def acceptance():
    """The fixed acceptance set of docs/perf-plan.md section 4 (fps only; counters come from the instrumented build)."""
    out = []
    for fmt in ("GRAY8", "GRAYS"):
        for opt in (0, 1):
            for mode in (-1, 0, 1):
                for th in (1, 4, 16):
                    out.append(S("cpu", R1080, fmt, "natural", [mode], opt=opt, threads=th))
    for fmt in ("GRAYS", "GRAY8"):
        for mode in (-1, 0, 1):
            for th in (1, 16):
                out.append(S("vulkan", R1080, fmt, "natural", [mode], threads=th))
        for sg in (0.0, 4.0):
            for th in (1, 16):
                out.append(S("vulkan", R1080, fmt, "natural", [0], sigma=sg, threads=th))
    for mode in (-1, 0, 1):
        for th in (1, 4, 16):
            out.append(S("cuda", R1080, "GRAYS", "natural", [mode], threads=th))
    out.append(S("vulkan", R1080, "GRAYS", "natural", [], shape="updown"))
    out.append(S("vulkan", R1080, "GRAY8", "natural", [], shape="updown"))
    for be in ("vulkan", "cuda"):
        for ch in ([1], [1, 1], [1, 1, 1, 1], [0]):
            out.append(S(be, R1080, "GRAYS", "natural", ch))
    out.append(S("vulkan", R1080, "GRAYS", "natural", [-1, -1, -1, 0]))
    return out

SUITES["acceptance"] = acceptance

def acceptance_cpu():
    out = []
    for fmt in ("GRAY8", "GRAYS"):
        for opt in (0, 1):
            for mode in (-1, 0, 1):
                for th in (1, 4, 8, 16):
                    out.append(S("cpu", R1080, fmt, "natural", [mode], opt=opt, threads=th))
    return out

SUITES["acceptance_cpu"] = acceptance_cpu

def acceptance_cuda():
    out = []
    for mode in (-1, 0, 1):
        for th in (1, 4, 16):
            out.append(S("cuda", R1080, "GRAYS", "natural", [mode], threads=th))
    for ch in ([1], [1, 1], [1, 1, 1, 1], [0]):
        out.append(S("cuda", R1080, "GRAYS", "natural", ch))
    return out

SUITES["acceptance_cuda"] = acceptance_cuda

def acceptance_upstream():
    """Same cells as acceptance_cpu, run against upstream TCanny r12 (load it with --plugin <libtcanny-r12.so> --plugin-force)."""
    out = []
    for fmt in ("GRAY8", "GRAYS"):
        for opt in (0, 1):
            for mode in (-1, 0, 1):
                for th in (1, 4, 8, 16):
                    out.append(S("upstream", R1080, fmt, "natural", [mode], opt=opt, threads=th))
    return out

SUITES["acceptance_upstream"] = acceptance_upstream

def acceptance_vulkan():
    out = []
    for fmt in ("GRAYS", "GRAY8"):
        for mode in (-1, 0, 1):
            for th in (1, 16):
                out.append(S("vulkan", R1080, fmt, "natural", [mode], threads=th))
        for sg in (0.0, 4.0):
            for th in (1, 16):
                out.append(S("vulkan", R1080, fmt, "natural", [0], sigma=sg, threads=th))
    out.append(S("vulkan", R1080, "GRAYS", "natural", [], shape="updown"))
    out.append(S("vulkan", R1080, "GRAY8", "natural", [], shape="updown"))
    for ch in ([1], [1, 1], [1, 1, 1, 1], [0], [-1, -1, -1, 0]):
        out.append(S("vulkan", R1080, "GRAYS", "natural", ch))
    return out

SUITES["acceptance_vulkan"] = acceptance_vulkan
