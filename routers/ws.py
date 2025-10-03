# Import các thư viện cần thiết cho WebSocket real-time
from __future__ import annotations

import json  # Xử lý JSON
from typing import Optional

import numpy as np  # Thư viện tính toán số học
from fastapi import APIRouter, WebSocket, WebSocketDisconnect  # WebSocket support

# Import các hàm xử lý từ module dsp và presets
from dsp import yin_pitch, apply_ema, apply_median
from presets import freq_to_note, nearest_target, PRESETS, note_to_freq
from settings import settings

# Tạo router cho WebSocket endpoints
router = APIRouter()


@router.websocket("/ws/pitch")
async def ws_pitch(
    websocket: WebSocket,  # Kết nối WebSocket
    preset: str = "guitar_standard",  # Preset nhạc cụ
    a4: float | None = None,  # Tần số chuẩn A4
    algo: str | None = None,  # Thuật toán phát hiện pitch
    smooth: str | None = None,  # Phương pháp làm mịn
    frame: int | None = None,  # Độ dài frame
    hop: int | None = None,  # Khoảng cách frame
    vad_rms: float | None = None,  # Ngưỡng VAD
    mode: str | None = None,  # Chế độ (auto/manual/chromatic)
    manual_note: Optional[str] = None,  # Nốt thủ công
):
    """
    WebSocket endpoint cho phân tích cao độ real-time
    
    Chức năng:
    1. Nhận dữ liệu âm thanh real-time từ client (Float32 PCM)
    2. Phát hiện cao độ sử dụng thuật toán YIN
    3. Áp dụng VAD và làm mịn dữ liệu
    4. Gửi kết quả phân tích real-time về client
    
    Client gửi: Float32 PCM audio data (mono, [-1, 1])
    Server trả về: JSON với thông tin cao độ, nốt nhạc, cents
    """
    # Chấp nhận kết nối WebSocket
    await websocket.accept()
    preset_key = preset if preset in PRESETS else "guitar_standard"
    string_set = PRESETS[preset_key].strings

    try:
        # Vòng lặp chính để xử lý dữ liệu real-time
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                # Chuyển đổi dữ liệu binary thành mảng numpy Float32
                # Dữ liệu âm thanh mono trong khoảng [-1, 1]
                audio = np.frombuffer(chunk, dtype=np.float32)

                # Sử dụng thuật toán YIN để phát hiện cao độ
                frame_len = frame or settings.frame_length
                hop_len = hop or settings.hop_length
                frames = yin_pitch(audio, sample_rate=settings.sample_rate, frame_length=frame_len, hop_length=hop_len, fmin_hz=30.0, fmax_hz=1200.0)

                # Lấy danh sách tần số cao độ từ frames
                f0_values = [float(pf.f0_hz) for pf in frames]
                
                # Áp dụng Voice Activity Detection (VAD) nếu được bật
                if vad_rms and vad_rms > 0:
                    window = np.hanning(frame_len).astype(np.float32)
                    rms_vals = []
                    for start in range(0, max(0, len(audio) - frame_len + 1), hop_len):
                        fr = audio[start : start + frame_len] * window
                        rms_vals.append(float(np.sqrt(np.mean(fr**2) + 1e-12)))
                    
                    # Kiểm tra mức âm thanh tổng thể
                    overall_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
                    if overall_rms < vad_rms:
                        # Nếu toàn bộ âm thanh quá yếu, loại bỏ tất cả frames
                        f0_values = [0.0] * len(f0_values)
                    else:
                        # Áp dụng VAD cho từng frame riêng biệt
                        for i, r in enumerate(rms_vals):
                            if r < vad_rms and i < len(f0_values):
                                f0_values[i] = 0.0
                
                # Làm mịn dữ liệu cao độ
                smoothing = (smooth or settings.smoothing).lower()
                if smoothing == "ema":
                    f0_values = apply_ema(f0_values, alpha=settings.ema_alpha)
                elif smoothing == "median":
                    f0_values = apply_median(f0_values, window=settings.median_window)

                # Xác định chế độ nhắm đích (targeting behavior)
                normalized_mode = (mode or "").lower()
                use_manual = normalized_mode == "manual" and (manual_note or "").strip() != ""
                manual_note_clean: Optional[str] = (manual_note or "").strip().upper() if use_manual else None

                # Xử lý từng frame và gửi kết quả real-time
                for pf, f0 in zip(frames, f0_values):
                    note, cents_from_note = freq_to_note(f0, a4_hz=a4 or settings.a4_hz)
                    
                    if normalized_mode == "chromatic":
                        # Chế độ chromatic: nốt đích = nốt phát hiện gần nhất
                        target_note = note
                        try:
                            target_freq = float(note_to_freq(target_note, a4_hz=a4 or settings.a4_hz)) if target_note else float("nan")
                        except Exception:
                            target_freq = float("nan")
                        cents_to_target = cents_from_note
                    elif use_manual and manual_note_clean is not None:
                        # Chế độ thủ công: khóa nốt đích theo nốt người dùng chỉ định
                        try:
                            target_freq_val = float(note_to_freq(manual_note_clean, a4_hz=a4 or settings.a4_hz))
                            target_note = manual_note_clean
                            # Tính cents đến nốt đích: dương nếu nốt đích cao hơn f0
                            cents_to_target = float(1200.0 * np.log2((target_freq_val if target_freq_val > 0 else 1.0) / (f0 if f0 > 0 else 1.0))) if f0 > 0 else float("nan")
                            target_freq = target_freq_val
                        except Exception:
                            # Fallback về chế độ tự động nếu có lỗi
                            target_note, target_freq, cents_to_target = nearest_target(f0, string_set)
                    else:
                        # Chế độ tự động: tìm nốt gần nhất trong preset
                        target_note, target_freq, cents_to_target = nearest_target(f0, string_set)
                    
                    # Tạo payload kết quả
                    payload = {
                        "time": pf.time_s,  # Thời gian (giây)
                        "f0_hz": f0,  # Tần số phát hiện (Hz)
                        "note": note,  # Nốt nhạc phát hiện
                        "cents_off": cents_from_note,  # Cents lệch so với nốt phát hiện
                        "target_note": target_note,  # Nốt đích
                        "target_freq": target_freq,  # Tần số nốt đích (Hz)
                        "cents_to_target": cents_to_target,  # Cents cần điều chỉnh
                    }
                    # Gửi kết quả về client qua WebSocket
                    await websocket.send_text(json.dumps(payload))
            else:
                # Bỏ qua text frames; client nên gửi Float32 PCM bytes
                pass
    except WebSocketDisconnect:
        # Xử lý khi client ngắt kết nối
        return


