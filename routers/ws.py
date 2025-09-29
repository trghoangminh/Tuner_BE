from __future__ import annotations

import json
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from dsp import autocorrelation_pitch, yin_pitch, apply_ema, apply_median
from presets import freq_to_note, nearest_target, PRESETS
from settings import settings


router = APIRouter()


@router.websocket("/ws/pitch")
async def ws_pitch(
    websocket: WebSocket,
    preset: str = "guitar_standard",
    a4: float | None = None,
    algo: str | None = None,
    smooth: str | None = None,
    frame: int | None = None,
    hop: int | None = None,
    vad_rms: float | None = None,
):
    await websocket.accept()
    preset_key = preset if preset in PRESETS else "guitar_standard"
    string_set = PRESETS[preset_key].strings

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                # Expect Float32 PCM (mono) in [-1, 1]
                audio = np.frombuffer(chunk, dtype=np.float32)

                algorithm = (algo or settings.algorithm).lower()
                frame_len = frame or settings.frame_length
                hop_len = hop or settings.hop_length
                if algorithm == "yin":
                    frames = yin_pitch(audio, sample_rate=settings.sample_rate, frame_length=frame_len, hop_length=hop_len)
                else:
                    frames = autocorrelation_pitch(audio, sample_rate=settings.sample_rate, frame_length=frame_len, hop_length=hop_len)

                f0_values = [float(pf.f0_hz) for pf in frames]
                if vad_rms and vad_rms > 0:
                    window = np.hanning(frame_len).astype(np.float32)
                    rms_vals = []
                    for start in range(0, max(0, len(audio) - frame_len + 1), hop_len):
                        fr = audio[start : start + frame_len] * window
                        rms_vals.append(float(np.sqrt(np.mean(fr**2) + 1e-12)))
                    for i, r in enumerate(rms_vals):
                        if r < vad_rms and i < len(f0_values):
                            f0_values[i] = 0.0
                smoothing = (smooth or settings.smoothing).lower()
                if smoothing == "ema":
                    f0_values = apply_ema(f0_values, alpha=settings.ema_alpha)
                elif smoothing == "median":
                    f0_values = apply_median(f0_values, window=settings.median_window)

                for pf, f0 in zip(frames, f0_values):
                    note, cents_from_note = freq_to_note(f0, a4_hz=a4 or settings.a4_hz)
                    target_note, target_freq, cents_to_target = nearest_target(f0, string_set)
                    payload = {
                        "time": pf.time_s,
                        "f0_hz": f0,
                        "note": note,
                        "cents_off": cents_from_note,
                        "target_note": target_note,
                        "target_freq": target_freq,
                        "cents_to_target": cents_to_target,
                    }
                    await websocket.send_text(json.dumps(payload))
            else:
                # Ignore text frames; client should send Float32 PCM bytes
                pass
    except WebSocketDisconnect:
        return


