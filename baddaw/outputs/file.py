"""FileOutput: the Output interface, captured to memory/WAV instead of a device.

This is how tests and agents "listen". The capture is one continuous stream
across plays, like a recording of the device output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .base import Output


class FileOutput(Output):
    """
    clock="offline": play() runs to completion immediately (loops stop after max_seconds).
    clock="manual":  time only moves when advance(n_frames) is called.
    """

    def __init__(self, path=None, clock: str = "offline", block: int = 512, max_seconds: float = 60.0):
        super().__init__()
        if clock not in ("offline", "manual"):
            raise ValueError("clock must be 'offline' or 'manual'")
        self.path = Path(path) if path is not None else None
        self.clock = clock
        self.block = block
        self.max_seconds = max_seconds
        self._blocks: list[np.ndarray] = []
        self._sr: int | None = None
        self._channels: int | None = None

    def captured(self) -> np.ndarray:
        """Everything output so far, shaped (frames, channels)."""
        if not self._blocks:
            return np.zeros((0, self._channels or 0), np.float32)
        return np.concatenate(self._blocks)

    def advance(self, frames: int) -> None:
        """Run the output clock forward. Idle time is not captured."""
        while frames > 0 and self.is_playing:
            n = min(frames, self.block)
            out = np.empty((n, self._channels), np.float32)
            written = self.pump.fill(out)
            self._blocks.append(out[:written])
            frames -= n
        self._flush()

    def _start(self):
        m = self.source.meta
        if self._sr is None:
            self._sr, self._channels = m.sr, m.channels
        elif (m.sr, m.channels) != (self._sr, self._channels):
            raise ValueError(
                f"capture is {self._channels}ch {self._sr}Hz; can't play {m.channels}ch {m.sr}Hz into it"
            )
        if self.clock == "offline":
            limit = round(self.max_seconds * m.sr) if self.loop else m.frames
            self.advance(limit)
            self.pump.finished = True

    def _stop(self):
        self._flush()

    def _flush(self):
        if self.path is not None and self._sr is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(self.path, self.captured(), self._sr, subtype="FLOAT")
