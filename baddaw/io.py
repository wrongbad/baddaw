"""Sources in, renders out."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import soundfile as sf

from .config import config
from .graph import Clip, as_signal, op
from .ops import Resample, Source
from .project import resolve


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load(path, name: str | None = None) -> Clip:
    """Reference an audio file. Reads the header and hashes the file; decoding is lazy."""
    p = resolve(path)
    info = sf.info(str(p))
    node = Source(
        path=str(p), sha256=_sha256(p), sr=info.samplerate,
        channels=info.channels, frames=info.frames, name=name or p.stem,
    )
    if info.samplerate != config.sample_rate:
        node = Resample(node, config.sample_rate)
    return Clip(node)


_PCM = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32"}


@op
def export(clip, path, subtype: str | None = None, engine=None) -> Path:
    """Render a clip to a file. Format follows the extension; wav/flac/aiff default to 24-bit."""
    node = as_signal(clip)
    p = resolve(path)
    if subtype is None and p.suffix.lower() in (".wav", ".flac", ".aif", ".aiff"):
        subtype = "PCM_24"
    data = (engine or config.engine).render(node).to_numpy()
    if subtype in _PCM and data.size and abs(data).max() > 1.0:
        warnings.warn(f"{node.label()} exceeds 0 dBFS and will clip in {subtype}", stacklevel=2)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), data, node.meta().sr, subtype=subtype)
    return p
