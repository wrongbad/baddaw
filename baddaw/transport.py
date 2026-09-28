"""Code-driven playback. One global transport: the configured output."""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass

from .config import config
from .graph import Clip, Signal, as_signal, op
from .outputs.base import RenderSource


@dataclass(frozen=True)
class Now:
    """What the transport last started. started_ms is wall-clock epoch ms."""

    node: Signal
    loop: bool
    started_ms: float


_now: Now | None = None

# observers (e.g. notebook views) notified on play/stop; weak so they never leak
_listeners = weakref.WeakSet()


def subscribe(listener) -> None:
    """listener.on_transport(now) with a Now on play, None on stop."""
    _listeners.add(listener)


def current() -> Now | None:
    return _now


def _notify():
    for listener in list(_listeners):
        listener.on_transport(_now)


class Playback:
    """Returned by play(). Prints as the transport status; in a notebook it displays
    the played clip's view with transport controls and a running playhead."""

    def __init__(self, clip: Clip):
        self.clip = clip

    def stop(self) -> None:
        stop()

    def __repr__(self):
        return status()

    def _repr_mimebundle_(self, include=None, exclude=None):
        from .display import widget_bundle
        return widget_bundle(self.clip)


@op
def play(clip, loop: bool = False) -> Playback:
    """Start playback and return immediately. Replaces anything already playing."""
    global _now
    node = as_signal(clip)
    config.output.play(RenderSource(config.engine, node), loop=loop)
    _now = Now(node, loop, time.time() * 1000)
    _notify()
    return Playback(Clip(node))


def stop() -> None:
    global _now
    if config._output is not None:
        config._output.stop()
    _now = None
    _notify()


def status() -> str:
    out = config._output
    if out is None or out.source is None:
        return "stopped"
    m = out.source.meta
    state = "playing" if out.is_playing else "stopped"
    pos = out.position() / m.sr
    loop = " loop" if out.loop else ""
    return f"{state} {out.source.node.label()} {pos:.2f}s / {m.duration:.2f}s{loop}"
