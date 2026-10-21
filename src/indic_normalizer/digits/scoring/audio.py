"""WAV loading and slicing (mono float32 at the model sample rate)."""

from __future__ import annotations

import os
import wave
from functools import lru_cache


def _np():
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError("audio scoring needs numpy: pip install 'indic-normalizer[digits]'") from e
    return np


def _read_wav_stdlib(path):
    np = _np()
    with wave.open(path) as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _resample(x, sr, target):
    if sr == target:
        return x
    try:
        import torch
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(x), sr, target).numpy()
    except ImportError:
        np = _np()
        n = int(round(len(x) * target / sr))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


@lru_cache(maxsize=4)
def load_wav(path: str, sample_rate: int = 16000):
    """Whole file as mono float32 numpy array at ``sample_rate`` (cached: a file's segments
    are scored together)."""
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except ImportError:
        x, sr = _read_wav_stdlib(path)
    return _resample(x, sr, sample_rate)


def slice_audio(x, start_s: float, end_s: float, sample_rate: int = 16000, pad_s: float = 0.0):
    a = max(0, int((start_s - pad_s) * sample_rate))
    b = min(len(x), int((end_s + pad_s) * sample_rate))
    return x[a:max(a + 1, b)]


def find_audio(audio_dir: str, stem: str) -> str:
    for ext in (".wav", ".WAV"):
        p = os.path.join(audio_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no {stem}.wav in {audio_dir}")
