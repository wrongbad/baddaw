from .base import BlockPump, Output, RenderSource
from .file import FileOutput


def __getattr__(name):
    # lazy: sounddevice is an optional dependency
    if name == "SoundDeviceOutput":
        from .sounddevice import SoundDeviceOutput
        return SoundDeviceOutput
    raise AttributeError(name)


__all__ = ["BlockPump", "FileOutput", "Output", "RenderSource", "SoundDeviceOutput"]
