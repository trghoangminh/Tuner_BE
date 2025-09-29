import asyncio
import io
import json
import wave
from pathlib import Path

import numpy as np
import requests
import websockets


API = "http://localhost:8000"
WS = "ws://localhost:8000/ws/pitch?preset=guitar_standard&algo=yin&smooth=ema"


def synth_tone_wav_bytes(freq_hz: float, seconds: float = 1.0, sr: int = 44100) -> bytes:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    x = 0.2 * np.sin(2 * np.pi * freq_hz * t)
    # 16-bit WAV for REST upload
    pcm16 = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def test_rest(freqs=(220.0, 440.0)):
    for f in freqs:
        data = synth_tone_wav_bytes(f)
        files = {"file": (f"tone_{int(f)}.wav", data, "audio/wav")}
        r = requests.post(f"{API}/analyze?preset=guitar_standard&algo=yin&smooth=ema", files=files)
        r.raise_for_status()
        js = r.json()
        med = js["summary"].get("median_f0_hz", 0.0)
        print(f"REST {f} Hz → median {med:.2f} Hz, frames={js['summary']['num_frames']}")


async def test_ws(freq_hz: float = 440.0, sr: int = 44100):
    # Send one Float32 chunk ~100ms
    t = np.arange(int(0.1 * sr), dtype=np.float32) / sr
    x = (0.2 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(x.tobytes())
        for _ in range(5):
            msg = await ws.recv()
            print("WS:", msg)


def main():
    print("Testing REST…")
    test_rest()
    print("Testing WS…")
    asyncio.run(test_ws(440.0))


if __name__ == "__main__":
    main()


