"""Passive notebook output: a text summary and a static waveform PNG."""

from __future__ import annotations

import struct
import zlib

import numpy as np


def _clock(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:05.2f}"


def text_summary(clip) -> str:
    m = clip.meta
    return f"Clip {clip.label()} {m.channels}ch {m.sr}Hz {_clock(m.duration)} ({m.frames} frames)"


_BG = (255, 255, 255)
_WAVE = (60, 60, 70)
_CLIP = (220, 40, 40)
_AXIS = (200, 200, 205)


def waveform_png(clip, width: int = 800, lane_height: int = 64) -> bytes:
    from .ops import peaks

    pk = np.asarray(peaks(clip, width))  # (cols, channels, 2)
    cols, channels = pk.shape[0], clip.channels
    h = lane_height * channels
    img = np.empty((h, width, 3), np.uint8)
    img[:] = _BG
    y = np.arange(lane_height)[:, None]
    for c in range(channels):
        lane = img[c * lane_height:(c + 1) * lane_height]
        lane[lane_height // 2, :] = _AXIS
        if cols == 0:
            continue
        lo, hi = np.clip(pk[:, c, 0], -1, 1), np.clip(pk[:, c, 1], -1, 1)
        # amplitude +1 at the top row, -1 at the bottom row
        top = np.round((1 - hi) / 2 * (lane_height - 1)).astype(int)
        bot = np.round((1 - lo) / 2 * (lane_height - 1)).astype(int)
        mask = (y >= top) & (y <= bot)  # (lane_height, cols)
        clipped = (pk[:, c, 1] >= 1) | (pk[:, c, 0] <= -1)
        color = np.where(clipped[:, None], _CLIP, _WAVE).astype(np.uint8)  # (cols, 3)
        region = lane[:, :cols]
        region[mask] = np.broadcast_to(color, (lane_height, cols, 3))[mask]
        if c:
            lane[0, :] = _AXIS
    return _encode_png(img)


def _encode_png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
        + chunk(b"IEND", b"")
    )
