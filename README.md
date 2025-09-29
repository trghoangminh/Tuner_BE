## Tuner_BE — Hướng dẫn cài đặt và chạy

Backend: FastAPI (Python) cho phân tích pitch theo file WAV và realtime qua WebSocket.

## Yêu cầu môi trường
- Python 3.10–3.12 (khuyến nghị)
- Windows PowerShell

## Cài đặt (Windows PowerShell)
```powershell
cd C:\Users\Minh\Desktop\Tuner_BE
py -m venv .venv
\.venv\Scripts\Activate.ps1

# Cập nhật công cụ build để tránh lỗi khi cài numpy/librosa
python -m pip install -U pip setuptools wheel

# Cài dependencies
pip install -r requirements.txt
```

Dependencies chính:
- fastapi, uvicorn[standard]
- numpy, librosa, soundfile
- python-multipart

## Chạy server
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Root: `http://localhost:8000/`
- API docs (Swagger): `http://localhost:8000/docs`

## Kiểm thử nhanh
- Health check
```powershell
curl http://localhost:8000/
```

- Phân tích file WAV
```powershell
curl -X POST http://localhost:8000/analyze -F "file=@C:\path\to\your.wav"
```
Kết quả trả về: `{ frames: [...], summary: {...} }`

### Script test tự động (220/440 Hz)
```powershell
# Server phải đang chạy tại http://localhost:8000
python scripts/test_tones.py
```
Script sẽ:
- Tạo WAV 220 Hz và 440 Hz, gửi lên `POST /analyze` với `algo=yin&smooth=ema` và in median f0.
- Mở WebSocket `ws://localhost:8000/ws/pitch` và gửi một chunk Float32 440 Hz (~100ms), in 5 phản hồi đầu.

## WebSocket realtime pitch
- Endpoint: `ws://localhost:8000/ws/pitch?preset=guitar_standard&a4=440&algo=yin&smooth=ema`
- Gửi bytes PCM Float32 (mono) ở sample rate 44100 Hz, server sẽ trả JSON từng frame:
```json
{
  "time": 0.023,
  "f0_hz": 439.8,
  "note": "A4",
  "cents_off": -2.1,
  "target_note": "A4",
  "target_freq": 440.0,
  "cents_to_target": 0.9
}
```

Tham số query (tuỳ chọn):
- `preset`: guitar_standard | ukulele_standard | violin_standard | viola_standard | cello_standard | bass_standard
- `a4`: 440 hoặc 442 (hoặc số bất kỳ)
- `algo`: `acf` (autocorrelation) hoặc `yin`
- `smooth`: `none`, `ema`, `median`

Ví dụ client Python gửi 1 chunk 440 Hz (Float32):
```python
import asyncio, websockets, numpy as np

async def main():
    uri = "ws://localhost:8000/ws/pitch?preset=guitar_standard"
    async with websockets.connect(uri, max_size=None) as ws:
        sr = 44100
        t = np.arange(0, 0.1, 1/sr, dtype=np.float32)
        x = (0.2*np.sin(2*np.pi*440*t)).astype(np.float32)
        pcm_float32 = x.tobytes()  # gửi Float32 PCM
        await ws.send(pcm_float32)
        for _ in range(5):
            print(await ws.recv())

asyncio.run(main())
```

## Cấu hình mặc định
Xem `settings.py`:
- `a4_hz = 440.0`
- `sample_rate = 44100`
- `hop_length = 512`
- `frame_length = 2048`
 - `algorithm = acf|yin` (mặc định `acf`)
 - `smoothing = none|ema|median` (mặc định `none`), `ema_alpha=0.3`, `median_window=5`

## Presets
Định nghĩa trong `presets.py` (ví dụ):
- `guitar_standard`: E2, A2, D3, G3, B3, E4
- `violin_standard`: G3, D4, A4, E5
- `ukulele_standard`: G4, C4, E4, A4

## Cấu trúc thư mục
```
Tuner_BE/
  main.py
  settings.py
  presets.py
  dsp.py
  requirements.txt
  routers/
    __init__.py
    analyze.py
    ws.py
```

## Lỗi cài đặt thường gặp
- Lỗi build `numpy/librosa`: hãy đảm bảo đã chạy `python -m pip install -U pip setuptools wheel` trước khi `pip install -r requirements.txt`.
- Sai phiên bản Python: dùng Python 3.10–3.12.


