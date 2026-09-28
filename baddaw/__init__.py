"""baddaw: a code-first DAW for Jupyter."""

from .config import config
from .engines import NumpyEngine
from .graph import Clip
from .io import export, load
from .ops import gain, peak_db, peaks, region, rms_db
from .outputs import FileOutput
from .transport import play, status, stop


def __getattr__(name):
    if name == "SoundDeviceOutput":
        from .outputs.sounddevice import SoundDeviceOutput
        return SoundDeviceOutput
    raise AttributeError(name)


__all__ = [
    "Clip", "FileOutput", "NumpyEngine", "SoundDeviceOutput", "config",
    "export", "gain", "load", "peak_db", "peaks", "play", "region", "rms_db",
    "status", "stop",
]
