"""Op definitions: node types plus the free functions that build them.

Each op is declared once here. Signal ops return new Clips (no computation).
Analysis ops build an Analysis node and ask the active engine to evaluate it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .graph import Analysis, Clip, Meta, Signal, as_signal, op


# ---- signal nodes ----

@dataclass(frozen=True)
class Source(Signal):
    """Decoded audio file. Identity is the file content hash, not its path."""

    path: str
    sha256: str
    sr: int
    channels: int
    frames: int
    name: str

    _unkeyed = ("path", "name")

    def meta(self):
        return Meta(self.sr, self.channels, self.frames)

    def label(self):
        return repr(self.name)


@dataclass(frozen=True)
class Resample(Signal):
    input: Signal
    sr: int

    def meta(self):
        m = self.input.meta()
        return Meta(self.sr, m.channels, round(m.frames * self.sr / m.sr))

    def label(self):
        return self.input.label()


@dataclass(frozen=True)
class Region(Signal):
    """Frames [start, stop) of the input."""

    input: Signal
    start: int
    stop: int

    def meta(self):
        m = self.input.meta()
        return Meta(m.sr, m.channels, self.stop - self.start)

    def label(self):
        sr = self.input.meta().sr
        return f"{self.input.label()}[{self.start / sr:.2f}:{self.stop / sr:.2f}]"


@dataclass(frozen=True)
class Gain(Signal):
    input: Signal
    db: float

    def meta(self):
        return self.input.meta()

    def label(self):
        return f"{self.input.label()}.gain({self.db:g})"


# ---- analysis nodes ----

@dataclass(frozen=True)
class Peak(Analysis):
    """Max absolute sample value over all channels (linear)."""

    input: Signal


@dataclass(frozen=True)
class Rms(Analysis):
    """RMS over all channels and frames (linear)."""

    input: Signal


@dataclass(frozen=True)
class Peaks(Analysis):
    """Waveform overview: array (width, channels, 2) of per-column (min, max)."""

    input: Signal
    width: int


# ---- free functions ----

def _to_db(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf


def _seconds_to_frame(t, sr, frames, default):
    if t is None:
        return default
    f = round(t * sr)
    if f < 0:
        f += frames
    return min(max(f, 0), frames)


@op
def region(clip, start: float | None = None, stop: float | None = None) -> Clip:
    """Time range in seconds. Negative times count from the end."""
    node = as_signal(clip)
    m = node.meta()
    a = _seconds_to_frame(start, m.sr, m.frames, 0)
    b = _seconds_to_frame(stop, m.sr, m.frames, m.frames)
    return Clip(Region(node, a, max(a, b)))


@op
def gain(clip, db: float) -> Clip:
    return Clip(Gain(as_signal(clip), float(db)))


@op
def peak_db(clip, engine=None) -> float:
    return _to_db(_evaluate(Peak(as_signal(clip)), engine))


@op
def rms_db(clip, engine=None) -> float:
    return _to_db(_evaluate(Rms(as_signal(clip)), engine))


@op
def peaks(clip, width: int = 800, engine=None):
    return _evaluate(Peaks(as_signal(clip), int(width)), engine)


def _evaluate(node, engine):
    from .config import config
    return (engine or config.engine).evaluate(node)
