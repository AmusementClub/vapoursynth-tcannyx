"""Per-case SHA-256 of TCanny CPU output at each `opt`, to prove a SIMD rewrite bit-identical to the build it replaces.

    python tests/cpu_simd_digest.py --plugin <old .so> --opt 0 2 3 --save base.json
    python tests/cpu_simd_digest.py --plugin <new .so> --opt 0 2 3 --compare base.json
    python tests/cpu_simd_digest.py --plugin <neon .so> --opt 0 --compare base.json --as 0=3   # NEON against x86 AVX2
    python tests/cpu_simd_digest.py --plugin <.so> --pinned                                    # meson test tcanny-cpu-simd

The corpus covers every operator and mode, the four blur branches (H+V, H only, V only, none), widths that are not a
multiple of any lane count, and planes at the minimum size. Cases that fail at creation hash their error text.
"""
import argparse
import hashlib
import json
import platform
import sys

import vapoursynth as vs

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vs_integration import noise_clip, planes_of  # noqa: E402

FORMATS = {"GRAY8": vs.GRAY8, "GRAY16": vs.GRAY16, "GRAYS": vs.GRAYS, "YUV420P8": vs.YUV420P8, "YUV444PS": vs.YUV444PS}
SIZES = ((132, 78), (37, 29), (13, 12), (6, 6), (3, 3))
SIGMAS = {"hv": dict(sigma=1.5), "h": dict(sigma=1.5, sigma_v=0.0), "v": dict(sigma=0.0, sigma_v=1.5), "none": dict(sigma=0.0)}


# Combined digests recorded from the VCL build that cpu_simd.cpp replaced (devhost, 2026-11-15). NEON uses FMA like AVX2
# and must equal it; SSE2 has no FMA. AVX-512 (opt=4) has no pinned value: no host to record it on.
PINNED = {
    "scalar": "a8cf5211a4dc42974975f2082ff74418c4bb39cd53c244bc335b238e693429eb",
    "sse2": "4cf86b26f5da29725b0b7fe0d074488039150278da09dae0e14ee086a9931ed4",
    "avx2": "338648175bfe2616bc9f8bada224c2b5ac0fdf68bacb0d141c935a533d59cdbc",
}


def combined(table):
    return hashlib.sha256("".join(table.values()).encode()).hexdigest()


def cases():
    for fname, fmt in FORMATS.items():
        for w, h in SIZES:
            if fmt == vs.YUV420P8:  # subsampled formats need even dimensions
                w, h = w + w % 2, h + h % 2
            for sname, sigma in SIGMAS.items():
                for mode in (-1, 0, 1):
                    for op in ((1,) if mode == -1 else range(7)):
                        if op == 5 and mode == 0:
                            continue
                        yield f"{fname}/{w}x{h}/{sname}/m{mode}/op{op}", fmt, w, h, dict(sigma, mode=mode, op=op)


def digests(core, opt):
    out = {}
    for key, fmt, w, h, kw in cases():
        src = noise_clip(core, fmt, w, h, 7, 2, 0.8)
        if core.get_video_format(fmt).sample_type == vs.FLOAT:
            kw = dict(kw, t_h=0.2, t_l=0.1)
        digest = hashlib.sha256()
        try:
            for f in core.tcannyx.TCanny(src, opt=opt, backend="cpu", **kw).frames():
                for p in planes_of(f):
                    digest.update(p.tobytes())
        except vs.Error as e:
            digest.update(b"error:" + str(e).encode())
        out[key] = digest.hexdigest()
    return out


def run_pinned(core) -> int:
    arm = platform.machine().lower() in ("arm64", "aarch64")
    checks = [(1, "scalar"), (0, "avx2")] if arm else [(1, "scalar"), (2, "sse2"), (3, "avx2")]
    probe = core.std.BlankClip(width=16, height=16, format=vs.GRAY8, length=1)
    failed = 0
    for opt, name in checks:
        try:
            core.tcannyx.TCanny(probe, opt=opt, backend="cpu").get_frame(0)
        except vs.Error as e:
            print(f"opt={opt} ({name}): skipped, {e}")
            continue
        got = combined(digests(core, opt))
        ok = got == PINNED[name]
        failed += not ok
        print(f"opt={opt} against pinned {name}: {'ok' if ok else 'MISMATCH ' + got}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--opt", type=int, nargs="+", default=[0, 2, 3])
    parser.add_argument("--pinned", action="store_true", help="check against the pinned VCL-era digests and exit")
    parser.add_argument("--save")
    parser.add_argument("--compare")
    parser.add_argument("--as", dest="mapping", nargs="*", default=[],
                        help="opt=baseline_opt pairs: compare this run's opt against another opt of the baseline")
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(args.plugin)
    if args.pinned:
        return run_pinned(core)
    result = {str(opt): digests(core, opt) for opt in args.opt}
    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=1, sort_keys=True)
    for opt, table in result.items():
        print(f"opt={opt}: {len(table)} cases, combined {combined(table)}")
    if not args.compare:
        return 0
    with open(args.compare) as f:
        baseline = json.load(f)
    mapping = dict(pair.split("=") for pair in args.mapping)
    failed = 0
    for opt, table in result.items():
        ref = baseline.get(mapping.get(opt, opt))
        if ref is None:
            print(f"opt={opt}: no baseline entry")
            failed += 1
            continue
        diff = [k for k in table if table[k] != ref.get(k)]
        print(f"opt={opt} vs baseline opt={mapping.get(opt, opt)}: {len(diff)} of {len(table)} cases differ")
        for k in diff[:40]:
            print("  ", k)
        failed += len(diff)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
