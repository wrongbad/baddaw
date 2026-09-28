"""Session-wide settings: project sample rate, root, active engine and output."""

from __future__ import annotations

from pathlib import Path


class Config:
    def __init__(self):
        self.sample_rate = 48000
        self.root: Path | None = None  # None -> auto-detect (see project.root)
        self._engine = None
        self._output = None

    @property
    def engine(self):
        if self._engine is None:
            from .engines import NumpyEngine
            self._engine = NumpyEngine()
        return self._engine

    @engine.setter
    def engine(self, value):
        self._engine = value

    @property
    def output(self):
        if self._output is None:
            from .outputs.sounddevice import SoundDeviceOutput
            self._output = SoundDeviceOutput()
        return self._output

    @output.setter
    def output(self, value):
        if self._output is not None and self._output is not value:
            self._output.stop()
        self._output = value


config = Config()
