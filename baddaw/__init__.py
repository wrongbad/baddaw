"""baddaw: a code-first DAW for Jupyter."""

from .config import config
from .display import view
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
    if name == "selection":  # needs the optional notebook ui
        from .ui import selection
        return selection
    raise AttributeError(name)


__all__ = [
    "Clip", "FileOutput", "NumpyEngine", "SoundDeviceOutput", "config",
    "export", "gain", "load", "peak_db", "peaks", "play", "region", "rms_db",
    "selection", "status", "stop", "view",
]
