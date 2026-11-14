#!/usr/bin/env python3
"""Vulkan smoke test: runs the filter once, checks it against the CPU scalar path for a few blur
radii, and checks that an unusable request fails with an error instead of crashing."""
import argparse
import json
import time

import numpy as np
import vapoursynth as vs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(args.plugin)
    clip = core.std.BlankClip(width=64, height=64, format=vs.GRAY8, length=1).std.GPUUpload()
    started = time.perf_counter()
    result = core.tcannyx.TCanny(clip, mode=-1, sigma=[0.0], backend="vulkan")
    result.get_frame(0)
    seconds = time.perf_counter() - started

    # Blur radius must follow the CPU (radius = max(int(sigma*3+0.5), 1), uncapped), incl. H-only / V-only.
    base = core.std.BlankClip(width=64, height=64, format=vs.GRAY8, length=1)

    def fill(n, f):
        frame = f.copy()
        plane = frame[0]
        array = np.ndarray((64, 64), dtype=np.uint8, buffer=plane, strides=(plane.strides[0], 1))
        y, x = np.indices(array.shape)
        array[:] = (x * 7 + y * 13 + (x * y) % 17) % 256
        return frame

    source = base.std.ModifyFrame([base], fill)
    for sigma_h, sigma_v in ((0.1, 0.1), (4.0, 4.0), (1.5, 0.0), (0.0, 1.5), (10.0, 10.0)):
        kw = dict(sigma=[sigma_h], sigma_v=[sigma_v], mode=-1)
        cpu = np.asarray(core.tcannyx.TCanny(source, opt=1, backend="cpu", **kw).get_frame(0)[0])
        gpu = np.asarray(core.tcannyx.TCanny(source.std.GPUUpload(), backend="vulkan", **kw).std.GPUDownload().get_frame(0)[0])
        if not np.array_equal(cpu, gpu):
            raise AssertionError(("blur radius mismatch", sigma_h, sigma_v, int(np.count_nonzero(cpu != gpu))))

    # A blur radius that does not fit the plane must fail cleanly (error, no crash, no leak).
    small = core.std.BlankClip(width=16, height=16, format=vs.GRAY8, length=1).std.GPUUpload()
    try:
        core.tcannyx.TCanny(small, sigma=[10.0], mode=-1, backend="vulkan").get_frame(0)
    except vs.Error:
        pass
    else:
        raise AssertionError("oversized blur radius was accepted")
    print(json.dumps({"backend": "vulkan", "status": "passed", "seconds": seconds}))


if __name__ == "__main__":
    main()
