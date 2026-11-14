#!/usr/bin/env python3
"""Every shader literal embedded in gpu_vulkan.cpp must be byte-identical to its TCanny/shaders/*.comp.glsl twin
(including the leading newline after R"glsl(). The plugin runs the literal; glslangValidator checks the file."""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
source = (root / "TCanny" / "gpu_vulkan.cpp").read_text()
twins = {"kShader": "tcanny_vulkan.comp.glsl", "kHysteresisShader": "tcanny_hysteresis.comp.glsl",
         "kPackShader": "tcanny_pack.comp.glsl", "kHysteresisTileShader": "tcanny_hysteresis_tile.comp.glsl"}
literals = dict(re.findall(r'constexpr char (k\w+Shader|kShader)\[\] = R"glsl\((.*?)\)glsl";', source, re.S))
status = 0
for name, file in twins.items():
    path = root / "TCanny" / "shaders" / file
    if name not in literals and not path.exists():
        continue  # shader not present in this revision
    if name not in literals or not path.exists():
        print(f"MISSING {name} / {file}")
        status = 1
    elif literals[name] != path.read_text():
        print(f"DIFFERS {name} vs {file}")
        status = 1
    else:
        print(f"ok {name} == {file}")
unknown = set(literals) - set(twins)
if unknown:
    print("literal without twin entry:", sorted(unknown))
    status = 1
sys.exit(status)
