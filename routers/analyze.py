# Import các thư viện cần thiết cho API phân tích âm thanh
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
import io  # Xử lý dữ liệu binary
import librosa  # Thư viện xử lý âm thanh
import numpy as np  # Thư viện tính toán số học
import soundfile as sf  # Đọc/ghi file âm thanh

# Import các hàm xử lý từ module dsp và presets
from dsp import yin_pitch, apply_ema, apply_median
from presets import freq_to_note, nearest_target, PRESETS, note_to_freq
from settings import settings

# Tạo router cho các endpoint phân tích âm thanh
router = APIRouter()


@router.post("")
async def analyze(
    file: UploadFile = File(...),  # File âm thanh được upload
    preset: str = Query("guitar_standard"),  # Preset nhạc cụ (guitar, violin, etc.)
    a4: float = Query(None),  # Tần số chuẩn của nốt A4 (mặc định 440Hz)
    algo: str = Query(None, regex="^(acf|yin)$"),  # Thuật toán phát hiện pitch
    sr: int = Query(None),  # Sample rate (tần số lấy mẫu)
    frame: int = Query(None),  # Độ dài frame để xử lý
    hop: int = Query(None),  # Khoảng cách giữa các frame
    smooth: str = Query(None, regex="^(none|ema|median)$"),  # Phương pháp làm mịn dữ liệu
    vad_rms: float = Query(0.0, description="RMS threshold for VAD; 0 to disable"),  # Ngưỡng VAD
    mode: str = Query(None, description="auto|manual"),  # Chế độ: tự động hoặc thủ công
    manual_note: str = Query(None, description="e.g., E2, A2, D3... when mode=manual"),  # Nốt thủ công
):
    """
    API endpoint phân tích file âm thanh để phát hiện cao độ (pitch)
    
    Chức năng chính:
    1. Nhận file WAV từ client
    2. Phát hiện cao độ sử dụng thuật toán YIN
    3. Áp dụng Voice Activity Detection (VAD) nếu cần
    4. Làm mịn dữ liệu cao độ
    5. Chuyển đổi tần số thành nốt nhạc và tính toán độ lệch
    6. Trả về kết quả phân tích chi tiết
    
    Returns:
        JSON chứa frames phân tích và summary thống kê
    """
    # Kiểm tra định dạng file - chỉ hỗ trợ WAV
    if file.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Đọc dữ liệu file từ request
    data = await file.read()
    try:
        # Đọc file WAV và chuyển đổi thành mảng numpy
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read WAV: {exc}")

    # Resample nếu sample rate khác với cài đặt mặc định
    if sr != settings.sample_rate:
        audio = librosa.resample(y=audio.astype(float), orig_sr=sr, target_sr=settings.sample_rate, res_type="kaiser_best")
        sr = settings.sample_rate

    # Sử dụng thuật toán YIN để phát hiện cao độ
    # YIN là thuật toán phát hiện pitch dựa trên tự tương quan
    frame_len = frame or settings.frame_length
    hop_len = hop or settings.hop_length
    frames = yin_pitch(audio, sample_rate=sr, frame_length=frame_len, hop_length=hop_len, fmin_hz=30.0, fmax_hz=1200.0)

    # Lấy preset nhạc cụ và danh sách nốt chuẩn
    preset_key = preset if preset in PRESETS else "guitar_standard"
    string_set = PRESETS[preset_key].strings

    # Voice Activity Detection (VAD) - loại bỏ các frame có âm thanh quá yếu
    if vad_rms > 0:
        hop_len = hop or settings.hop_length
        frame_len = frame or settings.frame_length
        
        # Kiểm tra mức âm thanh tổng thể trước
        overall_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        if overall_rms < vad_rms:
            # Nếu toàn bộ âm thanh quá yếu, loại bỏ tất cả frames
            for frame in frames:
                frame.f0_hz = 0.0
        else:
            # Áp dụng VAD cho từng frame riêng biệt
            rms_vals = []
            window = np.hanning(frame_len).astype(np.float32)
            for start in range(0, max(0, len(audio) - frame_len + 1), hop_len):
                fr = audio[start : start + frame_len]
                fr = fr * window  # Áp dụng cửa sổ Hanning
                rms = float(np.sqrt(np.mean(fr**2) + 1e-12))
                rms_vals.append(rms)
            
            # Loại bỏ cao độ cho các frame có RMS dưới ngưỡng
            for i, r in enumerate(rms_vals):
                if r < vad_rms and i < len(frames):
                    frames[i].f0_hz = 0.0

    # Làm mịn dữ liệu cao độ để giảm nhiễu
    f0_values = [float(pf.f0_hz) for pf in frames]
    smoothing = (smooth or settings.smoothing).lower()
    if smoothing == "ema":
        # Exponential Moving Average - phù hợp cho dữ liệu thay đổi liên tục
        f0_values = apply_ema(f0_values, alpha=settings.ema_alpha)
    elif smoothing == "median":
        # Median filter - tốt để loại bỏ các giá trị bất thường
        f0_values = apply_median(f0_values, window=settings.median_window)

    # Xử lý từng frame để tạo kết quả phân tích
    out_frames = []
    use_manual = (mode or "").lower() == "manual" and (manual_note or "").strip() != ""
    manual_note_clean = (manual_note or "").strip().upper() if use_manual else None
    
    for pf, f0 in zip(frames, f0_values):
        # Chuyển đổi tần số thành nốt nhạc và tính cents lệch
        note, cents_from_note = freq_to_note(f0, a4_hz=a4 or settings.a4_hz)
        
        # Xác định nốt đích (target) dựa trên chế độ
        if use_manual and manual_note_clean is not None:
            # Chế độ thủ công: sử dụng nốt do người dùng chỉ định
            try:
                target_freq_val = float(note_to_freq(manual_note_clean, a4_hz=a4 or settings.a4_hz))
                target_note = manual_note_clean
                # Tính cents từ tần số phát hiện đến nốt đích
                cents_to_target = float(1200.0 * np.log2((target_freq_val if target_freq_val > 0 else 1.0) / (f0 if f0 > 0 else 1.0))) if f0 > 0 else float("nan")
                target_freq = target_freq_val
            except Exception:
                # Fallback về chế độ tự động nếu có lỗi
                target_note, target_freq, cents_to_target = nearest_target(f0, string_set)
        else:
            # Chế độ tự động: tìm nốt gần nhất trong preset nhạc cụ
            target_note, target_freq, cents_to_target = nearest_target(f0, string_set)
        
        # Tạo frame kết quả với đầy đủ thông tin
        out_frames.append(
            {
                "time": pf.time_s,  # Thời gian (giây)
                "f0_hz": f0,  # Tần số phát hiện (Hz)
                "note": note,  # Nốt nhạc phát hiện
                "cents_off": cents_from_note,  # Cents lệch so với nốt phát hiện
                "target_note": target_note,  # Nốt đích cần điều chỉnh
                "target_freq": target_freq,  # Tần số nốt đích (Hz)
                "cents_to_target": cents_to_target,  # Cents cần điều chỉnh để đến nốt đích
            }
        )

    # Tạo thống kê tổng quan về kết quả phân tích
    nonzero = [f["f0_hz"] for f in out_frames if f["f0_hz"] > 0]  # Chỉ lấy các frame có cao độ
    algorithm = (algo or settings.algorithm)
    
    summary = {
        "median_f0_hz": float(np.median(nonzero)) if nonzero else 0.0,  # Tần số trung vị
        "num_frames": len(out_frames),  # Tổng số frame
        "preset": preset_key,  # Preset nhạc cụ sử dụng
        "algorithm": algorithm,  # Thuật toán phát hiện pitch
        "smoothing": smoothing,  # Phương pháp làm mịn
        "a4": float(a4 or settings.a4_hz),  # Tần số chuẩn A4
        "sample_rate": sr,  # Tần số lấy mẫu
        "frame_length": frame_len,  # Độ dài frame
        "hop_length": hop_len,  # Khoảng cách frame
    }

    # Trả về kết quả cuối cùng
    return {"frames": out_frames, "summary": summary}


