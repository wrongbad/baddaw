"""Live playback through PortAudio via the sounddevice package (optional extra `[play]`)."""

from __future__ import annotations

from .base import Output


class SoundDeviceOutput(Output):
    def __init__(self, device=None, blocksize: int = 0, latency="low"):
        super().__init__()
        import sounddevice  # noqa: F401  fail early with a clear ImportError
        self.device = device
        self.blocksize = blocksize
        self.latency = latency
        self._stream = None

    def _start(self):
        import sounddevice as sd

        pump = self.pump
        m = self.source.meta

        def callback(outdata, frames, time, status):
            pump.fill(outdata)
            if pump.finished:
                raise sd.CallbackStop

        self._stream = sd.OutputStream(
            samplerate=m.sr, channels=m.channels, dtype="float32",
            device=self.device, blocksize=self.blocksize, latency=self.latency,
            callback=callback,
        )
        self._stream.start()

    def _stop(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None
