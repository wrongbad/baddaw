"""Code-driven playback. One global transport: the configured output."""

from __future__ import annotations

from .config import config
from .graph import as_signal, op
from .outputs.base import RenderSource


@op
def play(clip, loop: bool = False) -> None:
    """Start playback and return immediately. Replaces anything already playing."""
    config.output.play(RenderSource(config.engine, as_signal(clip)), loop=loop)


def stop() -> None:
    if config._output is not None:
        config._output.stop()


def status() -> str:
    out = config._output
    if out is None or out.source is None:
        return "stopped"
    m = out.source.meta
    state = "playing" if out.is_playing else "stopped"
    pos = out.position() / m.sr
    loop = " loop" if out.loop else ""
    return f"{state} {out.source.node.label()} {pos:.2f}s / {m.duration:.2f}s{loop}"
