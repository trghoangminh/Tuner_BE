from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import io
import librosa
import numpy as np
import soundfile as sf

from dsp import autocorrelation_pitch, yin_pitch, apply_ema, apply_median
from presets import freq_to_note, nearest_target, PRESETS
from settings import settings


router = APIRouter()


@router.post("")
async def analyze(
    file: UploadFile = File(...),
    preset: str = Query("guitar_standard"),
    a4: float = Query(None),
    algo: str = Query(None, regex="^(acf|yin)$"),
    sr: int = Query(None),
    frame: int = Query(None),
    hop: int = Query(None),
    smooth: str = Query(None, regex="^(none|ema|median)$"),
    vad_rms: float = Query(0.0, description="RMS threshold for VAD; 0 to disable"),
):
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    data = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read WAV: {exc}")

    if sr != settings.sample_rate:
        audio = librosa.resample(y=audio.astype(float), orig_sr=sr, target_sr=settings.sample_rate, res_type="kaiser_best")
        sr = settings.sample_rate

    # Choose algorithm
    algorithm = (algo or settings.algorithm).lower()
    frame_len = frame or settings.frame_length
    hop_len = hop or settings.hop_length
    if algorithm == "yin":
        frames = yin_pitch(audio, sample_rate=sr, frame_length=frame_len, hop_length=hop_len)
    else:
        frames = autocorrelation_pitch(audio, sample_rate=sr, frame_length=frame_len, hop_length=hop_len)

    preset_key = preset if preset in PRESETS else "guitar_standard"
    string_set = PRESETS[preset_key].strings

    # Optional VAD: zero-out frames with low RMS
    if vad_rms > 0:
        hop_len = hop or settings.hop_length
        frame_len = frame or settings.frame_length
        # recompute frames for RMS check
        rms_vals = []
        window = np.hanning(frame_len).astype(np.float32)
        for start in range(0, max(0, len(audio) - frame_len + 1), hop_len):
            fr = audio[start : start + frame_len]
            fr = fr * window
            rms = float(np.sqrt(np.mean(fr**2) + 1e-12))
            rms_vals.append(rms)
        # zero f0 for rms below threshold
        for i, r in enumerate(rms_vals):
            if r < vad_rms and i < len(frames):
                frames[i].f0_hz = 0.0

    # Optional smoothing on f0 values
    f0_values = [float(pf.f0_hz) for pf in frames]
    smoothing = (smooth or settings.smoothing).lower()
    if smoothing == "ema":
        f0_values = apply_ema(f0_values, alpha=settings.ema_alpha)
    elif smoothing == "median":
        f0_values = apply_median(f0_values, window=settings.median_window)

    out_frames = []
    for pf, f0 in zip(frames, f0_values):
        note, cents_from_note = freq_to_note(f0, a4_hz=a4 or settings.a4_hz)
        target_note, target_freq, cents_to_target = nearest_target(f0, string_set)
        out_frames.append(
            {
                "time": pf.time_s,
                "f0_hz": f0,
                "note": note,
                "cents_off": cents_from_note,
                "target_note": target_note,
                "target_freq": target_freq,
                "cents_to_target": cents_to_target,
            }
        )

    nonzero = [f["f0_hz"] for f in out_frames if f["f0_hz"] > 0]
    summary = {
        "median_f0_hz": float(np.median(nonzero)) if nonzero else 0.0,
        "num_frames": len(out_frames),
        "preset": preset_key,
        "algorithm": algorithm,
        "smoothing": smoothing,
        "a4": float(a4 or settings.a4_hz),
        "sample_rate": sr,
        "frame_length": frame_len,
        "hop_length": hop_len,
    }

    return {"frames": out_frames, "summary": summary}


