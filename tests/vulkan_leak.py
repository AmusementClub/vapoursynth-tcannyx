#!/usr/bin/env python3
"""Leak check for the Vulkan backend: process many 1080p frames and watch device memory.

Per-frame scratch buffers are allocated from the core's pool and must be released once their
submission completes. nvidia-smi reports the whole device (the desktop included), so the check is
relative: after a warm-up, memory.used may fluctuate by a small margin but must not climb
frame after frame. Exits 1 if the growth after warm-up exceeds --max-growth-mb.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import vapoursynth as vs


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()[0]
    return int(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--max-growth-mb", type=int, default=64)
    parser.add_argument("--prefetch", type=int, default=4, help="concurrent frame requests (exercises the exec pool)")
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(args.plugin)
    core.max_cache_size = 256
    w, h = args.width, args.height
    y, x = np.indices((h, w))
    image = (((x * 13 + y * 17 + (x * y) % 29) % 256) / 255.0).astype(np.float32)
    base = core.std.BlankClip(width=w, height=h, format=vs.GRAYS, length=args.frames)

    def fill(n, f):
        frame = f.copy()
        plane = frame[0]
        array = np.ndarray((h, w), dtype=np.float32, buffer=plane, strides=(plane.strides[0], 4))
        array[:] = np.roll(image, n % w, axis=1)
        return frame

    source = base.std.ModifyFrame([base], fill).std.GPUUpload()
    report = {}
    status = 0
    for mode in (0, 1):
        clip = core.tcannyx.TCanny(source, sigma=[args.sigma], mode=mode, op=1, t_h=40.0, t_l=10.0,
                                  backend="vulkan").std.GPUDownload()
        samples = []
        for n, frame in enumerate(clip.frames(prefetch=args.prefetch)):
            if n % 50 == 0 or n == args.frames - 1:
                samples.append((n, vram_mib()))
            if n == args.warmup:
                baseline = vram_mib()
            del frame
        final = vram_mib()
        growth = final - baseline
        report[f"mode{mode}"] = {"samples_MiB": samples, "after_warmup_MiB": baseline, "final_MiB": final, "growth_MiB": growth}
        if growth > args.max_growth_mb:
            status = 1
    report["status"] = "failed" if status else "passed"
    print(json.dumps(report))
    sys.exit(status)


if __name__ == "__main__":
    main()
