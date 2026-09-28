"""Output interface and the shared block pump every output drives.

Outputs receive a RenderSource (engine + node + range), never raw arrays, so a
future compiled engine can hand them a native callback instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..engines.base import Engine
from ..graph import Meta, Signal


@dataclass(frozen=True)
class RenderSource:
    engine: Engine
    node: Signal
    start: int = 0
    stop: int | None = None

    @property
    def meta(self) -> Meta:
        m = self.node.meta()
        stop = m.frames if self.stop is None else self.stop
        return Meta(m.sr, m.channels, stop - self.start)

    def render(self) -> np.ndarray:
        return self.engine.render(self.node, self.start, self.stop).to_numpy()


class BlockPump:
    """Copies a pre-rendered range into output blocks, handling loop wrap and end.

    ``fill`` is the only thing an audio callback needs to call. It does no
    rendering and no allocation.
    """

    def __init__(self, data: np.ndarray, loop: bool = False):
        self.data = data
        self.loop = loop
        self.position = 0  # frame offset into data
        self.elapsed = 0  # frames output since start
        self.finished = len(data) == 0

    def fill(self, out: np.ndarray) -> int:
        """Fill ``out`` (frames, channels); zero-pads past the end. Returns frames of audio written."""
        n = len(out)
        written = 0
        while written < n and not self.finished:
            take = min(n - written, len(self.data) - self.position)
            out[written:written + take] = self.data[self.position:self.position + take]
            written += take
            self.position += take
            if self.position == len(self.data):
                if self.loop:
                    self.position = 0
                else:
                    self.finished = True
        out[written:] = 0
        self.elapsed += written
        return written


class Output(ABC):
    """Plays a RenderSource. One play at a time; a new play replaces the current one."""

    def __init__(self):
        self.source: RenderSource | None = None
        self.pump: BlockPump | None = None
        self.loop = False

    def play(self, source: RenderSource, loop: bool = False) -> None:
        self.stop()
        self.source = source
        self.loop = loop
        self.pump = BlockPump(source.render(), loop)
        self._start()

    def stop(self) -> None:
        if self.pump is not None:
            self._stop()
            self.pump.finished = True

    @property
    def is_playing(self) -> bool:
        return self.pump is not None and not self.pump.finished

    def position(self) -> int | None:
        """Frame offset into the playing range, or None if nothing was played."""
        return None if self.pump is None else self.pump.position

    @abstractmethod
    def _start(self) -> None:
        """Begin consuming self.pump."""

    @abstractmethod
    def _stop(self) -> None:
        """Stop consuming self.pump (called before it is marked finished)."""
