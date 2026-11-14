import argparse
import hashlib

import numpy as np
import vapoursynth as vs


EXPECTED = {
    ("GRAY8", -1, 1): "e1c572303f603dbb31ff2ab981a041aca97a1120635baf07b3db26a1f948b900",
    ("GRAY8", 0, 1): "15f3ed875289737b206ffeef52bc62f055d7e434b2cc81f9984994b6d378274a",
    ("GRAY8", 1, 1): "697081f5a274ac3c77ed3ef5451b058adf387be471d27089d0d8ab94e5785b37",
    ("GRAY16", 0, 1): "fd9243e1ba57263ed469c3bdbd7ade6ec5254e7ed924a9f5737fa44749933cc0",
    ("GRAYS", 1, 1): "c9682b0defaa771e8944163c36e835ceec77140010323366353f25b8c4aeeb84",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args()
    core = vs.core
    core.std.LoadPlugin(args.plugin)  # own namespace (tcannyx): an installed upstream TCanny does not conflict

    formats = [(vs.GRAY8, np.uint8), (vs.GRAY16, np.uint16), (vs.GRAYS, np.float32)]
    for fmt, dtype in formats:
        base = core.std.BlankClip(width=64, height=48, format=fmt, length=1)

        def fill(**kwargs):
            frame = kwargs["f"].copy()
            plane = frame[0]
            array = np.ndarray((48, 64), dtype=dtype, buffer=plane, strides=(plane.strides[0], dtype().nbytes))
            y, x = np.indices(array.shape)
            values = (x * 13 + y * 7 + (x * y) % 31) % 200
            array[:] = values if dtype != np.float32 else values / 255.0
            return frame

        source = base.std.ModifyFrame([base], fill)
        for mode in (-1, 0, 1):
            key = (fmt.name, mode, 1)
            if key not in EXPECTED:
                continue
            frame = core.tcannyx.TCanny(source, mode=mode, op=1, opt=1, backend="cpu").get_frame(0)
            digest = hashlib.sha256(np.asarray(frame[0]).copy().tobytes()).hexdigest()
            assert digest == EXPECTED[key], (key, digest, EXPECTED[key])


if __name__ == "__main__":
    main()

