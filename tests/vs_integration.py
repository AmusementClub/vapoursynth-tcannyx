import argparse

import numpy as np
import vapoursynth as vs


def expect_error(core, clip, text, **kwargs) -> None:
    """Creation (or first frame) must fail with one of the acceptable substrings."""
    texts = (text,) if isinstance(text, str) else tuple(text)
    try:
        core.tcannyx.TCanny(clip, **kwargs).get_frame(0)
    except vs.Error as e:
        assert any(t in str(e) for t in texts), (texts, str(e), kwargs)
        return
    raise AssertionError(f"expected error containing {text!r} for {kwargs}")


def run_ok(core, clip, **kwargs):
    frame = core.tcannyx.TCanny(clip, **kwargs).get_frame(0)
    assert frame.width == clip.width and frame.height == clip.height
    return frame


def blank(core, fmt, w, h):
    return core.std.BlankClip(width=w, height=h, format=fmt, length=1)


def run_small_plane_tests(core) -> None:
    # Gaussian radius for sigma=10 is 30; sigma=1.5 gives radius 5, so 6x6 is the minimum.
    expect_error(core, blank(core, vs.GRAY8, 16, 16), "too small", sigma=10.0)
    expect_error(core, blank(core, vs.GRAY8, 16, 16), "too small", sigma=10.0, mode=-1)
    expect_error(core, blank(core, vs.GRAY8, 64, 5), "vertical blur radius", sigma=1.5)
    expect_error(core, blank(core, vs.GRAY8, 5, 64), "horizontal blur radius", sigma=1.5)
    for mode in (-1, 0, 1):
        for opt in (0, 1):
            run_ok(core, blank(core, vs.GRAY8, 6, 6), sigma=1.5, mode=mode, opt=opt)
            run_ok(core, blank(core, vs.GRAYS, 6, 6), sigma=1.5, mode=mode, opt=opt)
    # Only processed planes are checked, and chroma is smaller than luma.
    expect_error(core, blank(core, vs.YUV420P8, 10, 10), "plane 1", sigma=[0.0, 1.5, 1.5])
    run_ok(core, blank(core, vs.YUV420P8, 12, 12), sigma=[0.0, 1.5, 1.5])
    run_ok(core, blank(core, vs.YUV420P8, 10, 10), sigma=[1.5, 10.0, 10.0], planes=[0])
    expect_error(core, blank(core, vs.YUV420P8, 10, 10), "plane 1", sigma=[1.5, 10.0, 10.0], planes=[0, 1])
    expect_error(core, blank(core, vs.YUV420P8, 4, 4), "too small", sigma=1.5)
    # Edge operator footprint: 3x3 for every operator; mode=-1 only blurs.
    for op in (0, 1, 2, 3, 4, 6):
        size = 3
        for mode in (0, 1):
            expect_error(core, blank(core, vs.GRAY8, size - 1, size - 1), "edge operator", sigma=0.0, op=op, mode=mode)
            expect_error(core, blank(core, vs.GRAY8, 16, size - 1), "edge operator", sigma=0.0, op=op, mode=mode)
            expect_error(core, blank(core, vs.GRAY8, size - 1, 16), "edge operator", sigma=0.0, op=op, mode=mode)
            for opt in (0, 1):
                run_ok(core, blank(core, vs.GRAY8, size, size), sigma=0.0, op=op, mode=mode, opt=opt)
                run_ok(core, blank(core, vs.GRAYS, size, size), sigma=0.0, op=op, mode=mode, opt=opt)
    expect_error(core, blank(core, vs.GRAY8, 2, 2), "edge operator", sigma=0.0, op=6, mode=0)
    run_ok(core, blank(core, vs.GRAY8, 2, 2), sigma=0.0, op=6, mode=-1)
    expect_error(core, blank(core, vs.GRAY8, 2, 2), "edge operator", sigma=0.0, op=5, mode=1)
    run_ok(core, blank(core, vs.GRAY8, 3, 3), sigma=0.0, op=5, mode=1)
    # Subsampled chroma below the operator footprint.
    expect_error(core, blank(core, vs.YUV420P8, 4, 4), "plane 1", sigma=0.0, op=6, mode=1)
    run_ok(core, blank(core, vs.YUV420P8, 6, 6), sigma=0.0, op=6, mode=1)


def run_backend_tests(core) -> None:
    # GPU backends on a CPU node must fail at creation. The exact message depends on the
    # build flavour (CPU-only / Vulkan / Vulkan+CUDA) and API level (<4.3 or >=4.3).
    api = "GPU backends require VapourSynth API 4.3"
    need_gpu_node = "GPU backend requires a GPU node"
    no_cuda = "backend=cuda requested but CUDA support was not compiled"
    clip = blank(core, vs.GRAYS, 64, 48)
    expect_error(core, clip, (api, no_cuda, need_gpu_node), backend="cuda")
    expect_error(core, clip, (api, need_gpu_node), backend="vulkan")
    expect_error(core, blank(core, vs.GRAY8, 64, 48), (api, no_cuda, need_gpu_node), backend="cuda")
    expect_error(core, clip, "backend must be", backend="bogus")
    run_ok(core, clip, backend="cpu")


def run_format_tests(core) -> None:
    for fmt in (vs.YUV420P8, vs.YUV444PS):
        clip = blank(core, fmt, 64, 48)
        for mode in (-1, 0, 1):
            frame = run_ok(core, clip, mode=mode, backend="cpu")
            assert frame.format.id == fmt


def noise_clip(core, fmt, w, h, seed, length, density):
    """Clip whose frames hold deterministic pseudo-random content: smooth blobs (strong edges) plus noise."""
    vf = core.get_video_format(fmt)
    blank = core.std.BlankClip(width=w, height=h, format=fmt, length=length)

    def fill(n, f):
        out = f.copy()
        rng = np.random.default_rng(seed * 1000 + n)
        for p in range(vf.num_planes):
            arr = np.asarray(out[p])
            ph, pw = arr.shape
            base = rng.random((ph // 8 + 2, pw // 8 + 2), dtype=np.float32)
            img = np.kron(base, np.ones((8, 8), np.float32))[:ph, :pw]
            img = img * density + rng.random((ph, pw), dtype=np.float32) * (1.0 - density)
            if vf.sample_type == vs.INTEGER:
                arr[:] = (img * ((1 << vf.bits_per_sample) - 1)).astype(arr.dtype)
            else:
                arr[:] = img
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def planes_of(frame):
    return [np.array(frame[p]) for p in range(frame.format.num_planes)]


def run_scratch_reuse_tests(core) -> None:
    """One instance sees alternating clips on several threads; every frame must equal a fresh instance's output."""
    old_threads = core.num_threads
    core.num_threads = 4
    try:
        for fmt in (vs.GRAY8, vs.GRAYS, vs.YUV420P8):
            a = noise_clip(core, fmt, 96, 80, 1, 6, 0.9)
            b = noise_clip(core, fmt, 96, 80, 2, 6, 0.3)
            mixed = core.std.Interleave([a, b])
            for mode in (-1, 0, 1):
                for opt in (0, 1):
                    kw = dict(sigma=1.5, mode=mode, opt=opt, t_h=0.2, t_l=0.1) if fmt != vs.GRAY8 and fmt != vs.YUV420P8 \
                        else dict(sigma=1.5, mode=mode, opt=opt)
                    shared = core.tcannyx.TCanny(mixed, **kw)
                    for n, frame in enumerate(shared.frames(prefetch=4)):
                        fresh_src = (a if n % 2 == 0 else b)[n // 2]
                        fresh = core.tcannyx.TCanny(fresh_src, **kw).get_frame(0)
                        for got, want in zip(planes_of(frame), planes_of(fresh)):
                            assert np.array_equal(got, want), (fmt, mode, opt, n)
    finally:
        core.num_threads = old_threads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args()

    core = vs.core
    core.std.LoadPlugin(args.plugin)  # own namespace (tcannyx): an installed upstream TCanny does not conflict
    clip = core.std.BlankClip(width=64, height=48, format=vs.GRAY8, length=2)
    for mode in (-1, 0, 1):
        output = core.tcannyx.TCanny(clip, sigma=[0.0], mode=mode, backend="auto")
        frame = output.get_frame(0)
        assert frame.width == 64 and frame.height == 48
        assert int(frame[0][0, 0]) == 0
    run_small_plane_tests(core)
    run_backend_tests(core)
    run_format_tests(core)
    run_scratch_reuse_tests(core)


if __name__ == "__main__":
    main()

