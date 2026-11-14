#!/usr/bin/env python3
import argparse
import json

import numpy as np
import vapoursynth as vs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--backend", choices=("vulkan", "cuda"), required=True)
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
                cpu = core.tcannyx.TCanny(source, sigma=[sigma], mode=mode, op=op,
                                         opt=1, backend="cpu").get_frame(0)
                gpu = core.tcannyx.TCanny(source.std.GPUUpload(), sigma=[sigma], mode=mode,
                                         op=op, backend=args.backend).std.GPUDownload().get_frame(0)
                a = np.asarray(cpu[0]).copy()
                b = np.asarray(gpu[0]).copy()
                delta = np.abs(a - b)
                # Operators read a 3x3 or 5x5 neighborhood. Exclude one
                # additional pixel from the core report because NMS compares
                # neighboring gradient values too.
                border = 3 if op == 6 else 2
                interior = delta[border:-border, border:-border]
                rows.append({
                    "sigma": sigma,
                    "op": op,
                    "mode": mode,
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                    "mismatch": int((delta > 2e-5).sum()),
                    "interior_mismatch": int((interior > 2e-5).sum()),
                })
    print(json.dumps({"backend": args.backend, "cases": rows}, separators=(",", ":")))


if __name__ == "__main__":
    main()
