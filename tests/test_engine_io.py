import math

import numpy as np
import pytest
import soundfile as sf

import baddaw as daw
from conftest import ramp


def render(clip):
    return daw.config.engine.render(clip.node).to_numpy()


def test_load_is_lazy(ramp_clip):
    clip, _ = ramp_clip
    assert clip.meta == daw.graph.Meta(48000, 2, 4800)
    assert daw.config.engine.misses == 0


def test_render_matches_file(ramp_clip):
    clip, data = ramp_clip
    np.testing.assert_array_equal(render(clip), data)


def test_render_range(ramp_clip):
    clip, data = ramp_clip
    buf = daw.config.engine.render(clip.node, 100, 250)
    assert (buf.frames, buf.channels, buf.sr) == (150, 2, 48000)
    np.testing.assert_array_equal(buf.to_numpy(), data[100:250])
    with pytest.raises(ValueError):
        daw.config.engine.render(clip.node, 0, 99999)


def test_buffers_are_read_only(ramp_clip):
    clip, _ = ramp_clip
    with pytest.raises(ValueError):
        render(clip)[0, 0] = 1.0


def test_region_and_gain(ramp_clip):
    clip, data = ramp_clip
    out = render(clip[0.01:0.05].gain(-6))
    np.testing.assert_allclose(out, data[480:2400] * 10 ** (-6 / 20), rtol=1e-6)


def test_cache_hit(ramp_clip):
    clip, _ = ramp_clip
    eng = daw.config.engine
    render(clip.gain(-3))
    misses = eng.misses
    render(daw.gain(clip, -3))  # structurally equal, new objects
    assert eng.misses == misses
    assert eng.hits >= 1


def test_resample_on_load(write_wav):
    sr = 44100
    t = np.arange(sr) / sr
    write_wav("a.wav", (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32), sr=sr)
    clip = daw.load("a.wav")
    assert (clip.sr, clip.frames, clip.channels) == (48000, 48000, 1)
    out = render(clip)
    assert out.shape == (48000, 1)
    # a 1 kHz tone survives resampling: amplitude and zero-crossing count
    assert math.isclose(daw.peak_db(clip), 20 * math.log10(0.5), abs_tol=0.1)
    crossings = np.count_nonzero(np.diff(np.signbit(out[1000:-1000, 0])))
    assert abs(crossings - 2 * 1000 * 46000 / 48000) <= 2


def test_measurements(write_wav):
    write_wav("dc.wav", np.full((1000, 2), 0.5, np.float32))
    clip = daw.load("dc.wav")
    assert math.isclose(clip.peak_db(), -6.0206, abs_tol=1e-3)
    assert math.isclose(clip.rms_db(), -6.0206, abs_tol=1e-3)
    assert clip[0.5:].peak_db() == -math.inf  # empty region


def test_peaks_overview(ramp_clip):
    clip, data = ramp_clip
    pk = clip.peaks(10)
    assert pk.shape == (10, 2, 2)
    np.testing.assert_array_equal(pk[0, :, 0], data[0])
    np.testing.assert_array_equal(pk[-1, :, 1], data[-1])
    assert clip[:0.00002].peaks(10).shape == (1, 2, 2)  # fewer frames than columns


def test_export_roundtrip(ramp_clip, tmp_path):
    clip, data = ramp_clip
    path = clip.gain(-6).export("out/quiet.wav")
    assert path == tmp_path / "out/quiet.wav"
    info = sf.info(path)
    assert (info.samplerate, info.channels, info.frames, info.subtype) == (48000, 2, 4800, "PCM_24")
    back, _ = sf.read(path, dtype="float32")
    np.testing.assert_allclose(back, data * 10 ** (-6 / 20), atol=2 ** -22)


def test_export_warns_on_clipping(write_wav):
    write_wav("hot.wav", np.full((100, 1), 0.9, np.float32))
    with pytest.warns(UserWarning, match="0 dBFS"):
        daw.load("hot.wav").gain(6).export("hot_out.wav")


def test_changed_source_is_detected(write_wav):
    write_wav("x.wav", ramp(100))
    clip = daw.load("x.wav")
    write_wav("x.wav", ramp(50))
    with pytest.raises(RuntimeError, match="changed on disk"):
        render(clip)


def test_png(ramp_clip):
    clip, _ = ramp_clip
    png = clip._repr_png_()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert int.from_bytes(png[16:20], "big") == 800  # width
    assert int.from_bytes(png[20:24], "big") == 128  # 2 lanes x 64
