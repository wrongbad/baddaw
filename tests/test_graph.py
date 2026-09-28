import pytest

import baddaw as daw
from baddaw.graph import Meta
from baddaw.ops import Gain, Region, Resample, Source


def src(sha="abc", path="/a.wav", name="a", sr=48000, frames=48000):
    return Source(path=path, sha256=sha, sr=sr, channels=2, frames=frames, name=name)


def test_building_nodes_does_no_rendering():
    eng = daw.config.engine
    clip = daw.Clip(src(path="/does/not/exist.wav"))
    out = clip.gain(-3)[0.25:0.5]
    assert eng.hits == eng.misses == 0
    assert out.meta == Meta(48000, 2, 12000)


def test_hash_is_structural():
    a = daw.gain(daw.Clip(src()), -3)
    b = daw.gain(daw.Clip(src()), -3)
    assert a.key == b.key
    assert a.key != daw.gain(daw.Clip(src()), -4).key
    assert a.key != daw.gain(daw.Clip(src(sha="def")), -3).key


def test_source_identity_ignores_path_and_name():
    assert src(path="/a.wav", name="a").key == src(path="/b/c.wav", name="c").key


def test_meta_inference():
    s = src(sr=44100, frames=44100)
    assert Resample(s, 48000).meta() == Meta(48000, 2, 48000)
    assert Region(s, 100, 300).meta() == Meta(44100, 2, 200)
    assert Gain(s, -1.0).meta() == s.meta()


def test_region_seconds_clamping_and_negative():
    c = daw.Clip(src())
    assert c[0.5:].frames == 24000
    assert c[:0.25].frames == 12000
    assert c[-0.25:].node.start == 36000
    assert c[0.9:5.0].frames == 4800
    assert c[0.8:0.2].frames == 0
    with pytest.raises(TypeError):
        c[0:1:2]


def test_clip_is_immutable_and_fluent():
    c = daw.Clip(src())
    with pytest.raises(AttributeError):
        c.node = None
    assert c.gain(-3).key == daw.gain(c, -3).key
    assert "gain" in dir(c)
    with pytest.raises(AttributeError):
        c.no_such_op


def test_labels():
    c = daw.Clip(src(name="take1", frames=5 * 48000))
    assert c[1.5:4.0].gain(-3).label() == "'take1'[1.50:4.00].gain(-3)"
    assert repr(c) == "Clip 'take1' 2ch 48000Hz 0:05.00 (240000 frames)"
