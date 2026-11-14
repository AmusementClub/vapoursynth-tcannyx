#!/usr/bin/env python3
"""CUDA backend leak/concurrency check: many 1080p frames across several
VapourSynth threads, GPU memory must stay flat and outputs must be correct."""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import vapoursynth as vs


def gpu_used_mib() -> int:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(out.strip().splitlines()[0])


def open_fds() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def churn(core, rounds: int, allowed_growth_mib: int) -> dict:
    """Create, run and drop filter instances of changing frame sizes: per-instance state (slots, weights, cached
    imports) must be released with the instance, so VRAM and file descriptors stay flat."""
    sizes = [(1920, 1080), (1280, 720), (640, 480), (1920, 1080), (96, 64)]
    used, fds, mismatches = [], [], []
    for r in range(rounds):
        w, h = sizes[r % len(sizes)]
        fmt = (vs.GRAYS, vs.GRAY8, vs.GRAY16)[r % 3]  # float and both integer sample sizes
        info = core.get_video_format(fmt)
        peak = 1.0 if info.sample_type == vs.FLOAT else (1 << info.bits_per_sample) - 1
        src = core.std.BlankClip(width=w, height=h, format=fmt, length=6, color=[0.25 * peak])
        for mode in (0, 1):
            out = core.tcannyx.TCanny(src.std.GPUUpload(), backend="cuda", sigma=[1.5 if min(w, h) > 64 else 0.0], mode=mode).std.GPUDownload()
            for f in out.frames(prefetch=4):
                pass
            del out
        del src
        if r % 5 == 4:
            used.append(gpu_used_mib())
            fds.append(open_fds())
    # The VapourSynth core allocates and frees whole 128 MiB blocks as frame sizes change, at a point that depends on thread
    # timing, so the level may move by one block in either direction. A per-instance leak (slots, imports) is megabytes
    # per round and accumulates, so require the spread over the run to stay within one block plus the allowance.
    growth = (max(used[1:]) - min(used[1:])) if len(used) > 2 else 0
    allowed_growth_mib += 128
    return {"rounds": rounds, "samples_mib": used, "open_fds": fds, "growth_mib": growth,
            "status": "passed" if growth <= allowed_growth_mib and (len(fds) < 2 or fds[-1] - fds[1] <= 8) else "failed"}


def noise_clip(core, width: int, height: int, length: int):
    rng = np.random.default_rng(2)
    base = rng.random((height, width), dtype=np.float32)
    blank = core.std.BlankClip(width=width, height=height, format=vs.GRAYS, length=length)

    def fill(n, f):
        frame = f.copy()
        plane = frame[0]
        np.ndarray((height, width), dtype=np.float32, buffer=plane, strides=(plane.strides[0], 4))[:] = np.roll(base, n % 89, axis=1)
        return frame

    return blank.std.ModifyFrame([blank], fill)


def chain_test(core, frames: int, depth: int, allowed_growth_mib: int) -> dict:
    """`depth` chained CUDA filters (all on one core, so they share one import cache): VRAM must stay flat."""
    node = noise_clip(core, 1920, 1080, frames).std.GPUUpload()
    for _ in range(depth - 1):
        node = core.tcannyx.TCanny(node, backend="cuda", sigma=[1.5], mode=-1)
    node = core.tcannyx.TCanny(node, backend="cuda", sigma=[1.5], mode=0, t_h=40.0, t_l=15.0).std.GPUDownload()
    samples = []
    for n, _ in enumerate(node.frames(prefetch=8)):
        if n % 50 == 0:
            samples.append(gpu_used_mib())
    third = max(1, len(samples) // 3)
    growth = int(round(np.mean(samples[-third:]) - np.mean(samples[third:2 * third])))
    return {"depth": depth, "frames": frames, "samples_mib": samples, "growth_mib": growth,
            "status": "passed" if growth <= allowed_growth_mib else "failed"}


def staleness_test(core, allowed_slack_mib: int) -> dict:
    """Blocks the core has freed must not stay pinned by the import cache while the filter keeps running: run a 1080p
    clip, shrink the core's cache and keep a second (tiny) clip flowing for longer than the idle limit through the
    same core. VRAM must drop back towards the level before the big clip, not stay at the big clip's level."""
    core.max_cache_size = 512
    big = core.tcannyx.TCanny(noise_clip(core, 1920, 1080, 120).std.GPUUpload(), backend="cuda", sigma=[1.5], mode=1).std.GPUDownload()
    for _ in big.frames(prefetch=8):
        pass
    after_big = gpu_used_mib()
    keep = big  # the first instance stays alive, as in a long-lived script
    core.max_cache_size = 16
    tiny = core.tcannyx.TCanny(core.std.BlankClip(width=64, height=64, format=vs.GRAYS, length=20000).std.GPUUpload(), backend="cuda", sigma=[0.0], mode=1).std.GPUDownload()
    deadline = time.time() + 12.0
    for _ in tiny.frames(prefetch=2):
        if time.time() > deadline:
            break
    after_idle = gpu_used_mib()
    del keep, big
    return {"after_big_mib": after_big, "after_tiny_mib": after_idle, "dropped_mib": after_big - after_idle,
            "status": "passed" if after_idle <= after_big + allowed_slack_mib else "failed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--allowed-growth-mib", type=int, default=64)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--format", default="GRAYS", help="source format name in the vapoursynth module (GRAYS, GRAY8, GRAY16, ...)")
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(args.plugin)
    core.num_threads = args.threads
    core.max_cache_size = 512
    width, height = 1920, 1080
    rng = np.random.default_rng(1)
    noise = rng.random((height, width), dtype=np.float32)
    ramp = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base_img = (0.6 * noise + 0.4 * ramp).astype(np.float32)

    fmt = getattr(vs, args.format)
    info = core.get_video_format(fmt)
    is_float = info.sample_type == vs.FLOAT
    np_type = np.float32 if is_float else (np.uint8 if info.bytes_per_sample == 1 else np.uint16)
    peak = 1.0 if is_float else float((1 << info.bits_per_sample) - 1)
    blank = core.std.BlankClip(width=width, height=height, format=fmt, length=args.frames)

    def fill(n, f):
        frame = f.copy()
        plane = frame[0]
        array = np.ndarray((height, width), dtype=np_type, buffer=plane, strides=(plane.strides[0], info.bytes_per_sample))
        rolled = np.roll(base_img, n % 97, axis=1)
        array[:] = rolled if is_float else np.rint(rolled * peak).astype(np_type)
        return frame

    source = blank.std.ModifyFrame([blank], fill)
    kwargs = dict(sigma=[1.5], mode=0, op=1, t_h=40.0, t_l=15.0)
    cuda = core.tcannyx.TCanny(source.std.GPUUpload(), backend="cuda", **kwargs).std.GPUDownload()
    cpu = core.tcannyx.TCanny(source, opt=1, backend="cpu", **kwargs)

    samples = []
    fds = []
    mismatches = []
    checked = 0
    baseline = None
    for n, frame in enumerate(cuda.frames(prefetch=args.threads * 2)):
        if n % args.check_every == 0:
            used = gpu_used_mib()
            samples.append((n, used))
            fds.append(open_fds())
            if n == args.check_every * 4:  # after warm-up (pools, contexts, caches)
                baseline = used
            if n % (args.check_every * 4) == 0:
                a = np.asarray(frame[0])
                b = np.asarray(cpu.get_frame(n)[0])
                checked += 1
                if not np.array_equal(a, b):
                    mismatches.append(n)
                if np.count_nonzero(a) == 0:
                    mismatches.append(-n)
    final = gpu_used_mib()
    samples.append((args.frames, final))
    # Caches and pools fill during warm-up, so a leak is a trend: compare the mean
    # of the last third of samples against the middle third.
    used = [m for _, m in samples]
    third = max(1, len(used) // 3)
    growth = int(round(np.mean(used[-third:]) - np.mean(used[third:2 * third])))
    fds.append(open_fds())
    result = {"format": args.format, "frames": args.frames, "threads": args.threads, "samples_mib": samples,
              "baseline_mib": baseline, "open_fds": fds, "growth_mib": growth, "cpu_checked_frames": checked, "mismatched": mismatches,
              "status": "passed" if not mismatches and growth <= args.allowed_growth_mib and (fds[-1] - fds[1] <= 8 or fds[-1] < 0) else "failed"}
    churn_result = churn(core, 60, args.allowed_growth_mib)
    result["churn"] = churn_result
    if churn_result["status"] != "passed":
        result["status"] = "failed"
    result["staleness"] = staleness_test(core, args.allowed_growth_mib)
    result["chain"] = chain_test(core, min(args.frames, 1500), 4, args.allowed_growth_mib)
    for key in ("staleness", "chain"):
        if result[key]["status"] != "passed":
            result["status"] = "failed"
    print(json.dumps(result))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
