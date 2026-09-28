"""Headless v0 walkthrough: load, inspect, play (to file), export.

    python examples/v0/v0.py [path/to/recording.wav]

Without an argument, a synthetic take is generated at 44.1 kHz so the resample
path is exercised. Everything is written next to this script (flat project layout).
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import baddaw as daw

here = Path(__file__).parent
daw.config.root = here

if len(sys.argv) > 1:
    path = Path(sys.argv[1]).resolve()
else:
    sr = 44100
    t = np.arange(3 * sr) / sr
    tone = 0.4 * np.sin(2 * np.pi * 220 * t) * np.exp(-2 * (t % 1))
    noise = 0.05 * np.random.default_rng(0).standard_normal(len(t))
    path = here / "demo.wav"
    sf.write(path, np.stack([tone + noise, tone - noise], axis=1), sr)

take = daw.load(path)
print(take)
print(f"peak {take.peak_db():.1f} dBFS, rms {take.rms_db():.1f} dBFS")

(here / "take.png").write_bytes(take._repr_png_())

# play a looped slice into a capture file instead of a device
daw.config.output = daw.FileOutput(here / "capture.wav", max_seconds=2.0)
loop = take[0.5:1.0]
daw.play(loop, loop=True)
print(daw.status())

quiet = take.gain(-6)
out = quiet.export("quiet.wav")
print(f"exported {out.relative_to(here)}")

# check what we "heard" and wrote
cap, cap_sr = sf.read(here / "capture.wav", dtype="float32")
expected = daw.config.engine.render(loop.node).to_numpy()
assert cap_sr == take.sr and len(cap) == 2 * take.sr
assert np.array_equal(cap[: loop.frames], expected), "capture doesn't match the loop"
assert np.array_equal(cap[loop.frames : 2 * loop.frames], expected), "loop didn't wrap"

back = daw.load(out)
assert back.frames == take.frames
assert abs(back.peak_db() - (take.peak_db() - 6)) < 0.01
print("ok")
