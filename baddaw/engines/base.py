"""Engine and Buffer interfaces.

An engine turns graph nodes into audio (``render``) or values (``evaluate``).
Buffers are engine-owned; ``to_numpy`` is the single interop point with the
outside world (file writers, audio devices, display).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np

from ..graph import Analysis, Signal


class Buffer(Protocol):
    sr: int
    channels: int
    frames: int

    def to_numpy(self) -> np.ndarray:
        """Read-only float32 array shaped (frames, channels)."""
        ...


class Engine(ABC):
    @abstractmethod
    def render(self, node: Signal, start: int = 0, stop: int | None = None) -> Buffer:
        """Render frames [start, stop) of a signal node. Length always matches the range."""

    @abstractmethod
    def evaluate(self, node: Analysis) -> Any:
        """Compute an analysis node's value."""
