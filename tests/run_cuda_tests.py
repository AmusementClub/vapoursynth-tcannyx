#!/usr/bin/env python3
import argparse
import json
import time

import vapoursynth as vs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(args.plugin)
    clip = core.std.BlankClip(width=64, height=64, format=vs.GRAYS, length=1).std.GPUUpload()
    started = time.perf_counter()
    try:
        result = core.tcannyx.TCanny(clip, mode=-1, sigma=[0.0], backend="cuda")
        result.get_frame(0)
    except Exception as exc:
        print(json.dumps({"backend": "cuda", "status": "unavailable",
                          "reason": "CUDA runtime could not access a usable driver/device from this process",
                          "error": str(exc)}))
        raise SystemExit(2)
    print(json.dumps({"backend": "cuda", "status": "passed", "seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()

