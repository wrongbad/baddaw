import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

import baddaw as daw
from baddaw.outputs import BlockPump


def pump_all(pump, block, total):
    outs = []
    for _ in range(0, total, block):
        out = np.empty((block, pump.data.shape[1]), np.float32)
        pump.fill(out)
        outs.append(out)
    return np.concatenate(outs)[:total]


def test_pump_uneven_blocks():
    data = np.arange(10, dtype=np.float32)[:, None]
    out = pump_all(BlockPump(data), block=4, total=12)
    np.testing.assert_array_equal(out[:10], data)
    np.testing.assert_array_equal(out[10:], 0)


def test_pump_loop_wrap_inside_block():
    data = np.arange(3, dtype=np.float32)[:, None]
    pump = BlockPump(data, loop=True)
    out = pump_all(pump, block=8, total=8)
    np.testing.assert_array_equal(out[:, 0], [0, 1, 2, 0, 1, 2, 0, 1])
    assert pump.position == 2 and pump.elapsed == 8 and not pump.finished


def test_pump_empty():
    pump = BlockPump(np.zeros((0, 2), np.float32), loop=True)
    assert pump.finished
    out = np.ones((4, 2), np.float32)
    assert pump.fill(out) == 0
    np.testing.assert_array_equal(out, 0)


def test_offline_play_captures_region(ramp_clip, tmp_path):
    clip, data = ramp_clip
    daw.config.output = out = daw.FileOutput(tmp_path / "cap.wav")
    daw.play(clip[0.01:0.02])
    np.testing.assert_array_equal(out.captured(), data[480:960])
    back, sr = sf.read(tmp_path / "cap.wav", dtype="float32")
    np.testing.assert_array_equal(back, data[480:960])
    assert daw.status() == "stopped 'ramp'[0.01:0.02] 0.01s / 0.01s"


def test_offline_loop_is_tiled(ramp_clip):
    clip, data = ramp_clip
    daw.config.output = out = daw.FileOutput(max_seconds=0.025)  # 1200 frames = 2.5 loops
    clip[0.0:0.01].play(loop=True)
    np.testing.assert_array_equal(out.captured(), np.concatenate([data[:480]] * 3)[:1200])


def test_manual_clock_position_and_status(ramp_clip):
    clip, data = ramp_clip
    daw.config.output = out = daw.FileOutput(clock="manual", block=100)
    daw.play(clip[0.01:0.03], loop=True)
    assert out.is_playing and len(out.captured()) == 0
    out.advance(250)
    assert out.position() == 250
    assert daw.status() == "playing 'ramp'[0.01:0.03] 0.01s / 0.02s loop"
    out.advance(960)  # wraps once
    assert out.position() == 250
    np.testing.assert_array_equal(out.captured()[:960], data[480:1440])


def test_stop_truncates_capture(ramp_clip):
    clip, data = ramp_clip
    daw.config.output = out = daw.FileOutput(clock="manual")
    daw.play(clip)
    out.advance(300)
    daw.stop()
    out.advance(300)
    assert not out.is_playing
    np.testing.assert_array_equal(out.captured(), data[:300])
    assert daw.status().startswith("stopped")


def test_new_play_replaces_current(ramp_clip):
    clip, data = ramp_clip
    daw.config.output = out = daw.FileOutput(clock="manual")
    daw.play(clip)
    out.advance(200)
    daw.play(clip[0.05:])
    out.advance(200)
    np.testing.assert_array_equal(out.captured(), np.concatenate([data[:200], data[2400:2600]]))


def test_capture_format_is_fixed(ramp_clip, write_wav):
    clip, _ = ramp_clip
    write_wav("mono.wav", np.zeros((10, 1), np.float32))
    daw.play(clip)
    with pytest.raises(ValueError, match="2ch"):
        daw.play(daw.load("mono.wav"))


def test_import_without_sounddevice():
    code = (
        "import sys; sys.modules['sounddevice'] = None\n"
        "import baddaw\n"
        "try:\n"
        "    baddaw.SoundDeviceOutput()\n"
        "except ImportError:\n"
        "    print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"
