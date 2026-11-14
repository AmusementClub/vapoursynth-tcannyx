"""Print one SHA-256 over TCanny CPU output for every opt/mode/format; compare the line between builds
(for example a normal build against one compiled with -DTCANNY_POISON_SCRATCH=2)."""
import argparse
import hashlib
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vs_integration import noise_clip, planes_of  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--plugin", required=True)
args = parser.parse_args()
core = vs.core
core.std.LoadPlugin(args.plugin)
h = hashlib.sha256()
for fmt in (vs.GRAY8, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV444PS):
    src = noise_clip(core, fmt, 132, 78, 3, 3, 0.8)
    for opt in (1, 2, 3):
        for mode in (-1, 0, 1):
            for sigma in (0.0, 1.5):
                try:
                    out = core.tcannyx.TCanny(src, sigma=sigma, mode=mode, opt=opt)
                    for f in out.frames():
                        for p in planes_of(f):
                            h.update(p.tobytes())
                except vs.Error as e:
                    h.update(str(e).encode())
print(h.hexdigest())
