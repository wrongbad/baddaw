"""Lazy signal graph: immutable, hashable nodes and the user-facing Clip handle.

Nothing in this module touches sample data. Nodes describe *what* to compute;
engines (see ``baddaw.engines``) decide *how*.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Meta:
    """Static shape of a signal, known without rendering."""

    sr: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sr


@dataclass(frozen=True)
class Node:
    """Base for all graph nodes. Subclasses are frozen dataclasses."""

    # fields excluded from the structural key (e.g. display names, local paths)
    _unkeyed = ()

    def inputs(self) -> tuple[Node, ...]:
        return tuple(
            v for f in dataclasses.fields(self)
            if isinstance(v := getattr(self, f.name), Node)
        )

    @functools.cached_property
    def key(self) -> str:
        """Structural hash: op type + params + input keys."""
        h = hashlib.sha256(type(self).__qualname__.encode())
        for f in dataclasses.fields(self):
            if f.name in self._unkeyed:
                continue
            v = getattr(self, f.name)
            part = v.key if isinstance(v, Node) else repr(v)
            h.update(f"|{f.name}={part}".encode())
        return h.hexdigest()


class Signal(Node):
    """A node that produces audio."""

    def meta(self) -> Meta:
        raise NotImplementedError

    def label(self) -> str:
        """Short human/agent-readable description for status lines."""
        raise NotImplementedError


class Analysis(Node):
    """A node that produces a value (float, array) derived from a signal."""


# free-function op registry; Clip exposes these as fluent methods
OPS: dict[str, Callable[..., Any]] = {}


def op(fn):
    OPS[fn.__name__] = fn
    return fn


def as_signal(x) -> Signal:
    if isinstance(x, Clip):
        return x.node
    if isinstance(x, Signal):
        return x
    raise TypeError(f"expected Clip or Signal, got {type(x).__name__}")


class Clip:
    """Immutable handle to a signal node.

    Holds no samples and computes nothing; every method builds a new node.
    """

    __slots__ = ("node",)

    def __init__(self, node: Signal):
        object.__setattr__(self, "node", node)

    def __setattr__(self, name, value):
        raise AttributeError("Clip is immutable")

    @property
    def meta(self) -> Meta:
        return self.node.meta()

    @property
    def sr(self) -> int:
        return self.meta.sr

    @property
    def channels(self) -> int:
        return self.meta.channels

    @property
    def frames(self) -> int:
        return self.meta.frames

    @property
    def duration(self) -> float:
        return self.meta.duration

    @property
    def key(self) -> str:
        return self.node.key

    def label(self) -> str:
        return self.node.label()

    def __getitem__(self, s):
        if not isinstance(s, slice) or s.step is not None:
            raise TypeError("Clip supports time slices in seconds, e.g. clip[1.5:4.0]")
        return OPS["region"](self, s.start, s.stop)

    def __getattr__(self, name):
        try:
            fn = OPS[name]
        except KeyError:
            raise AttributeError(name) from None
        return functools.partial(fn, self)

    def __dir__(self):
        return [*super().__dir__(), *OPS]

    def __repr__(self):
        from .display import text_summary
        return text_summary(self)

    def _repr_png_(self):
        from .display import waveform_png
        return waveform_png(self)
