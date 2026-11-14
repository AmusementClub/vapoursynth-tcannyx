#!/usr/bin/env python3
import argparse
import json

import numpy as np
import vapoursynth as vs


def metrics(reference: np.ndarray, actual: np.ndarray, border: int) -> dict:
    delta = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    core = delta[border:-border, border:-border]
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.percentile(delta, 99)),
        "mismatch_1e-5": int((delta > 1e-5).sum()),
        "mismatch_2e-5": int((delta > 2e-5).sum()),
        "core_mismatch_2e-5": int((core > 2e-5).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--backend", choices=("vulkan", "cuda"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(args.plugin)
    width, height = 128, 96
    base = core.std.BlankClip(width=width, height=height, format=vs.GRAYS, length=1)

    def fill(**kwargs):
        frame = kwargs["f"].copy()
        plane = frame[0]
        array = np.ndarray((height, width), dtype=np.float32, buffer=plane,
                           strides=(plane.strides[0], 4))
        y, x = np.indices(array.shape)
        array[:] = ((x * 13 + y * 17 + (x * y) % 29) % 256) / 255.0
        return frame

    source = base.std.ModifyFrame([base], fill)
    rows = []
    for sigma in (0.0, 1.5):
        for op in range(7):
            for mode in (-1, 0, 1):
                if op == 5 and mode == 0:
                    continue
                scalar = np.asarray(core.tcannyx.TCanny(
                    source, sigma=[sigma], mode=mode, op=op, opt=1,
                    backend="cpu").get_frame(0)[0]).copy()
                simd = np.asarray(core.tcannyx.TCanny(
                    source, sigma=[sigma], mode=mode, op=op, opt=0,
                    backend="cpu").get_frame(0)[0]).copy()
                gpu = np.asarray(core.tcannyx.TCanny(
                    source.std.GPUUpload(), sigma=[sigma], mode=mode, op=op,
                    backend=args.backend).std.GPUDownload().get_frame(0)[0]).copy()
                border = 3 if op == 6 else 2
                row = {"sigma": sigma, "op": op, "mode": mode,
                       "cpu_simd_vs_scalar": metrics(scalar, simd, border),
                       "gpu_vs_scalar": metrics(scalar, gpu, border)}
                if mode == 0:
                    row["gpu_false_positive"] = int(((gpu > 0.5) & (scalar <= 0.5)).sum())
                    row["gpu_false_negative"] = int(((gpu <= 0.5) & (scalar > 0.5)).sum())
                rows.append(row)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"backend": args.backend, "width": width, "height": height,
                   "cases": rows}, handle, indent=2)
    print(json.dumps({"backend": args.backend, "cases": len(rows), "output": args.output}))


if __name__ == "__main__":
    main()
