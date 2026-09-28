import numpy as np
import pytest
import soundfile as sf

import baddaw as daw


@pytest.fixture(autouse=True)
def fresh_config(tmp_path):
    """Isolate every test: fresh engine, in-memory file output, tmp project root."""
    daw.config.sample_rate = 48000
    daw.config.root = tmp_path
    daw.config.engine = daw.NumpyEngine()
    daw.config.output = daw.FileOutput()
    yield daw.config
    daw.config.output.stop()


def ramp(frames, channels=2):
    """Distinct, easily checked samples: channel c at frame i = (i + c * frames) / big."""
    x = np.arange(frames * channels, dtype=np.float64).reshape(channels, frames).T
    return (x / (frames * channels)).astype(np.float32)


@pytest.fixture
def write_wav(tmp_path):
    def write(name, data, sr=48000):
        path = tmp_path / name
        sf.write(path, data, sr, subtype="FLOAT")
        return path
    return write


@pytest.fixture
def ramp_clip(write_wav):
    data = ramp(4800)
    write_wav("ramp.wav", data)
    return daw.load("ramp.wav"), data
