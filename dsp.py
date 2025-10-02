from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import librosa

from settings import settings


@dataclass
class PitchFrame:
    time_s: float
    f0_hz: float


def yin_pitch(
    audio: np.ndarray,
    sample_rate: int | None = None,
    frame_length: int | None = None,
    hop_length: int | None = None,
    fmin_hz: float = 30.0,
    fmax_hz: float = 1200.0,
) -> List[PitchFrame]:
    sr = sample_rate or settings.sample_rate
    n_fft = frame_length or settings.frame_length
    hop = hop_length or settings.hop_length

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return []

    fmin = max(1.0, fmin_hz)
    fmax = max(fmin + 1.0, fmax_hz)

    f0 = librosa.yin(
        y=audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop,
        trough_threshold=0.1,
    )
    frames: List[PitchFrame] = []
    for i, hz in enumerate(f0):
        hz_val = float(hz) if np.isfinite(hz) else 0.0
        frames.append(PitchFrame(time_s=(i * hop) / sr, f0_hz=hz_val))
    return frames


def apply_ema(values: List[float], alpha: float) -> List[float]:
    if not values:
        return []
    smoothed: List[float] = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(float(ema))
    return smoothed


def apply_median(values: List[float], window: int) -> List[float]:
    if not values or window <= 1:
        return list(values)
    half = window // 2
    padded = [values[0]] * half + list(values) + [values[-1]] * half
    out: List[float] = []
    for i in range(len(values)):
        out.append(float(np.median(padded[i : i + window])))
    return out


def frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> List[float]:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    window = np.hanning(frame_length).astype(np.float32)
    out: List[float] = []
    for start in range(0, max(0, len(audio) - frame_length + 1), hop_length):
        fr = audio[start : start + frame_length] * window
        out.append(float(np.sqrt(np.mean(fr**2) + 1e-12)))
    return out



