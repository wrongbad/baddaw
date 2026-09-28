"""Reference engine: whole-node numpy renders with an LRU cache keyed by node hash."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr

from ..graph import Analysis, Signal
from ..ops import Gain, Peak, Peaks, Region, Resample, Rms, Source
from .base import Engine

_KERNELS = {}


def kernel(node_type):
    def register(fn):
        _KERNELS[node_type] = fn
        return fn
    return register


@dataclass(frozen=True)
class NumpyBuffer:
    data: np.ndarray  # (frames, channels) float32, read-only
    sr: int

    @property
    def channels(self) -> int:
        return self.data.shape[1]

    @property
    def frames(self) -> int:
        return self.data.shape[0]

    def to_numpy(self) -> np.ndarray:
        return self.data


class NumpyEngine(Engine):
    def __init__(self, cache_bytes: int = 1 << 30):
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cached_bytes = 0
        self.hits = 0
        self.misses = 0

    def render(self, node: Signal, start: int = 0, stop: int | None = None) -> NumpyBuffer:
        m = node.meta()
        stop = m.frames if stop is None else stop
        if not 0 <= start <= stop <= m.frames:
            raise ValueError(f"range [{start}, {stop}) outside 0..{m.frames}")
        return NumpyBuffer(self._full(node)[start:stop], m.sr)

    def evaluate(self, node: Analysis):
        return _KERNELS[type(node)](self, node)

    def _full(self, node: Signal) -> np.ndarray:
        key = node.key
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        m = node.meta()
        data = np.ascontiguousarray(_KERNELS[type(node)](self, node), dtype=np.float32)
        if data.shape != (m.frames, m.channels):
            raise RuntimeError(f"{type(node).__name__} rendered {data.shape}, meta says {(m.frames, m.channels)}")
        data.setflags(write=False)
        self._store(key, data)
        return data

    def _store(self, key: str, data: np.ndarray):
        self._cache[key] = data
        self._cached_bytes += data.nbytes
        while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cached_bytes -= old.nbytes


def _fit(x: np.ndarray, frames: int) -> np.ndarray:
    if len(x) >= frames:
        return x[:frames]
    return np.pad(x, ((0, frames - len(x)), (0, 0)))


@kernel(Source)
def _source(eng, node: Source):
    data, sr = sf.read(node.path, dtype="float32", always_2d=True)
    if sr != node.sr or data.shape != (node.frames, node.channels):
        raise RuntimeError(f"{node.path} changed on disk since it was loaded")
    return data


@kernel(Resample)
def _resample(eng, node: Resample):
    x = eng._full(node.input)
    y = soxr.resample(x, node.input.meta().sr, node.sr, quality="VHQ")
    return _fit(y.reshape(-1, x.shape[1]), node.meta().frames)


@kernel(Region)
def _region(eng, node: Region):
    return eng._full(node.input)[node.start:node.stop]


@kernel(Gain)
def _gain(eng, node: Gain):
    return eng._full(node.input) * np.float32(10 ** (node.db / 20))


@kernel(Peak)
def _peak(eng, node: Peak):
    x = eng._full(node.input)
    return float(np.max(np.abs(x))) if x.size else 0.0


@kernel(Rms)
def _rms(eng, node: Rms):
    x = eng._full(node.input)
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


@kernel(Peaks)
def _peaks(eng, node: Peaks):
    x = eng._full(node.input)
    frames, channels = x.shape
    width = min(node.width, frames)
    if width == 0:
        return np.zeros((0, channels, 2), np.float32)
    # frames >= width, so bin edges are strictly increasing
    edges = (np.arange(width) * frames) // width
    lo = np.minimum.reduceat(x, edges, axis=0)
    hi = np.maximum.reduceat(x, edges, axis=0)
    return np.stack([lo, hi], axis=-1)
