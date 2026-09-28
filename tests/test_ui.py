import time

import numpy as np
import pytest

import baddaw as daw

pytest.importorskip("anywidget")
from baddaw.ui import ClipView  # noqa: E402


@pytest.fixture
def view(ramp_clip):
    clip, _ = ramp_clip
    v = ClipView(clip)
    v.sent = []
    v.send = lambda content, buffers=None: v.sent.append(content)
    return v


def browser(v, **msg):
    v._on_msg(v, msg, [])


def test_view_state(view, ramp_clip):
    clip, data = ramp_clip
    assert (view.sr, view.frames, view.channels) == (48000, 4800, 2)
    pk = np.frombuffer(view.peaks, np.float32).reshape(view.columns, 2, 2)
    np.testing.assert_array_equal(pk[0, :, 0], data[0])
    assert repr(view) == f"ClipView({clip!r})"


def test_selection_round_trips_to_code(view, ramp_clip):
    clip, _ = ramp_clip
    assert view.range is None and view.region() is clip
    view.selection = [0.01, 0.025]  # as the browser would sync it
    assert view.range == (0.01, 0.025)
    assert daw.selection() == (0.01, 0.025)
    # the selection is the same node you'd get by typing the slice
    assert view.region().key == clip[0.01:0.025].key


def playing(view):
    p = dict(view.playing)
    assert time.time() * 1000 - p.pop("t0") < 5000
    return p


def test_browser_play_uses_kernel_transport(view, ramp_clip):
    _, data = ramp_clip
    daw.config.output = out = daw.FileOutput()
    browser(view, cmd="play", start=0.01, stop=0.02, loop=False)
    np.testing.assert_array_equal(out.captured(), data[480:960])
    assert playing(view) == {"start": 480, "stop": 960, "loop": False}


def test_browser_play_from_cursor_to_end(view, ramp_clip):
    _, data = ramp_clip
    daw.config.output = out = daw.FileOutput()
    browser(view, cmd="play", start=0.09, stop=None, loop=False)
    np.testing.assert_array_equal(out.captured(), data[4320:])


def test_code_play_moves_view_playhead(view, ramp_clip):
    clip, _ = ramp_clip
    daw.play(clip, loop=True)
    assert playing(view) == {"start": 0, "stop": 4800, "loop": True}
    daw.play(clip[0.05:0.06])
    assert playing(view) == {"start": 2400, "stop": 2880, "loop": False}
    daw.stop()
    assert view.playing is None


def test_playing_other_clip_stops_view_playhead(view, ramp_clip):
    clip, _ = ramp_clip
    daw.play(clip)
    daw.play(clip.gain(-3))
    assert view.playing is None


def test_view_created_after_play_shows_playhead(ramp_clip):
    clip, _ = ramp_clip
    daw.play(clip[0.05:0.06], loop=True)
    assert playing(ClipView(clip)) == {"start": 2400, "stop": 2880, "loop": True}
    assert playing(ClipView(clip[0.05:0.06])) == {"start": 0, "stop": 480, "loop": True}


def test_play_returns_handle_with_controls(ramp_clip, monkeypatch):
    clip, _ = ramp_clip
    daw.config.output = daw.FileOutput(clock="manual")
    pb = daw.play(clip[0.01:0.02], loop=True)
    assert repr(pb) == "playing 'ramp'[0.01:0.02] 0.00s / 0.01s loop"
    assert pb._repr_mimebundle_() is None  # headless: just the status text
    monkeypatch.setattr("baddaw.display._in_kernel", lambda: True)
    assert "application/vnd.jupyter.widget-view+json" in pb._repr_mimebundle_()
    pb.stop()
    assert repr(pb).startswith("stopped")


def test_errors_are_shown_in_view(view):
    browser(view, cmd="play", start="x", stop=None, loop=False)
    assert view.sent[-1]["event"] == "error"


def test_no_widget_outside_kernel(ramp_clip):
    clip, _ = ramp_clip
    assert clip._repr_mimebundle_() is None


def test_widget_bundle_in_kernel(ramp_clip, monkeypatch):
    clip, _ = ramp_clip
    monkeypatch.setattr("baddaw.display._in_kernel", lambda: True)
    bundle = clip._repr_mimebundle_()
    assert list(bundle) == ["application/vnd.jupyter.widget-view+json"]
    assert bundle["application/vnd.jupyter.widget-view+json"]["model_id"]
    daw.config.widgets = False
    try:
        assert clip._repr_mimebundle_() is None
    finally:
        daw.config.widgets = True
