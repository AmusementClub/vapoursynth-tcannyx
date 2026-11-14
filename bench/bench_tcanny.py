#!/usr/bin/env python3
import argparse
import json
import time

import vapoursynth as vs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--backend", default="cpu")
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(args.plugin)
    fmt = vs.GRAYS if args.backend == "cuda" else vs.GRAY8
    clip = core.std.BlankClip(width=1920, height=1080, format=fmt, length=args.frames)
    if args.backend != "cpu":
        clip = clip.std.GPUUpload()
    output = core.tcannyx.TCanny(clip, mode=1, sigma=[1.5], backend=args.backend)
    if args.backend != "cpu":
        # Force the producer timeline to complete so GPU numbers include the
        # actual kernel and the explicit graph boundary.
        output = output.std.GPUDownload()
    started = time.perf_counter()
    for n in range(args.frames):
        output.get_frame(n)
    elapsed = time.perf_counter() - started
    print(json.dumps({"backend": args.backend, "frames": args.frames, "seconds": elapsed,
                      "fps": args.frames / elapsed if elapsed else 0.0}))


if __name__ == "__main__":
    main()

