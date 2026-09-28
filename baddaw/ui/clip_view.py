"""Interactive notebook view of a Clip: waveform, playhead, cursor, range selection.

A remote control only. Playback runs in the kernel through the same transport
as daw.play(), and the selection is readable from Python (view.selection).
"""

from __future__ import annotations

import re
from pathlib import Path

import anywidget
import numpy as np
import traitlets as t

from .. import transport
from ..graph import Clip, as_signal
from ..ops import Region, peaks, region

# waveform detail sent to the browser; the view reduces it to pixel width
COLUMNS = 4096

_HERE = Path(__file__).parent


def bundle_esm() -> str:
    """anywidget loads one module, so inline the pure core (which node tests import directly)."""
    core = (_HERE / "clip_view_core.js").read_text()
    view = (_HERE / "clip_view.js").read_text()
    view = re.sub(r'^import \{[^}]*\} from "\./clip_view_core\.js";\n', "", view, flags=re.M)
    return core + "\n" + view


_last_selected: ClipView | None = None


class ClipView(anywidget.AnyWidget):
    _esm = bundle_esm()
    _css = _HERE / "clip_view.css"

    label = t.Unicode().tag(sync=True)
    sr = t.Int().tag(sync=True)
    frames = t.Int().tag(sync=True)
    channels = t.Int().tag(sync=True)
    columns = t.Int().tag(sync=True)
    peaks = t.Bytes().tag(sync=True)  # float32 (columns, channels, 2)
    # [start, stop] in seconds, snapped to 1 ms in the view; [] when nothing is selected
    selection = t.List(t.Float(), default_value=[]).tag(sync=True)
    loop = t.Bool(False).tag(sync=True)
    # what the kernel is playing within this clip: {start, stop (frames), loop, t0 (epoch ms)}
    # or None. Synced state rather than messages, so views rendered later still see it.
    playing = t.Dict(allow_none=True, default_value=None).tag(sync=True)

    def __init__(self, clip: Clip):
        self.clip = clip
        node = as_signal(clip)
        m = node.meta()
        pk = np.ascontiguousarray(peaks(clip, COLUMNS), dtype=np.float32)
        super().__init__(
            label=node.label(), sr=m.sr, frames=m.frames, channels=m.channels,
            columns=pk.shape[0], peaks=pk.tobytes(),
        )
        self.on_msg(self._on_msg)
        self.observe(self._on_selection, "selection")
        transport.subscribe(self)
        self.on_transport(transport.current())

    # ---- python api ----

    @property
    def range(self) -> tuple[float, float] | None:
        """Selected (start, stop) in seconds, or None."""
        return tuple(self.selection) if len(self.selection) == 2 else None

    def region(self) -> Clip:
        """The selected part of the clip (whole clip if nothing is selected)."""
        return region(self.clip, *self.range) if self.range else self.clip

    def __repr__(self):
        return f"ClipView({self.clip!r})"

    # ---- plumbing ----

    def _on_selection(self, change):
        global _last_selected
        _last_selected = self

    def _on_msg(self, _, msg, buffers):
        try:
            if msg["cmd"] == "play":
                transport.play(region(self.clip, msg["start"], msg.get("stop")), loop=bool(msg.get("loop")))
            elif msg["cmd"] == "stop":
                transport.stop()
        except Exception as e:  # surface in the view; comm handler errors are otherwise invisible
            self.send({"event": "error", "message": f"{type(e).__name__}: {e}"})

    def on_transport(self, now):
        """Show the playhead when this clip (or a region of it) is played from anywhere."""
        self.playing = _within(as_signal(self.clip), now)


def _within(mine, now):
    if now is None:
        return None
    node = now.node
    if node.key == mine.key:
        start, stop = 0, node.meta().frames
    elif isinstance(node, Region) and node.input.key == mine.key:
        start, stop = node.start, node.stop
    else:
        return None
    return {"start": start, "stop": stop, "loop": now.loop, "t0": now.started_ms}


def view(clip) -> ClipView:
    return ClipView(clip)


def selection() -> tuple[float, float] | None:
    """(start, stop) seconds of the most recently changed selection in any view."""
    return _last_selected.range if _last_selected is not None else None
