# Import các thư viện cần thiết cho xử lý tín hiệu âm thanh
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np  # Thư viện tính toán số học
import librosa  # Thư viện xử lý âm thanh

from settings import settings  # Import cấu hình từ file settings


@dataclass
class PitchFrame:
    """Lớp lưu trữ thông tin về cao độ (pitch) tại một thời điểm cụ thể"""
    time_s: float  # Thời gian tính bằng giây
    f0_hz: float   # Tần số cơ bản (fundamental frequency) tính bằng Hz


def yin_pitch(
    audio: np.ndarray,
    sample_rate: int | None = None,
    frame_length: int | None = None,
    hop_length: int | None = None,
    fmin_hz: float = 30.0,
    fmax_hz: float = 1200.0,
) -> List[PitchFrame]:
    """
    Hàm phát hiện cao độ (pitch) sử dụng thuật toán YIN
    
    Args:
        audio: Mảng âm thanh đầu vào
        sample_rate: Tần số lấy mẫu (Hz)
        frame_length: Độ dài frame để xử lý
        hop_length: Khoảng cách giữa các frame
        fmin_hz: Tần số tối thiểu để tìm kiếm (Hz)
        fmax_hz: Tần số tối đa để tìm kiếm (Hz)
    
    Returns:
        Danh sách các PitchFrame chứa thông tin cao độ theo thời gian
    """
    # Sử dụng giá trị mặc định từ settings nếu không được cung cấp
    sr = sample_rate or settings.sample_rate
    n_fft = frame_length or settings.frame_length
    hop = hop_length or settings.hop_length

    # Chuyển đổi âm thanh stereo thành mono nếu cần
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32)
    
    # Kiểm tra nếu âm thanh rỗng
    if audio.size == 0:
        return []

    # Đảm bảo fmin và fmax hợp lệ
    fmin = max(1.0, fmin_hz)
    fmax = max(fmin + 1.0, fmax_hz)

    # Sử dụng thuật toán YIN để phát hiện cao độ
    # YIN là thuật toán phát hiện pitch dựa trên tự tương quan
    f0 = librosa.yin(
        y=audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop,
        trough_threshold=0.1,  # Ngưỡng để xác định đỉnh trong tự tương quan
    )
    
    # Chuyển đổi kết quả thành danh sách PitchFrame
    frames: List[PitchFrame] = []
    for i, hz in enumerate(f0):
        # Xử lý giá trị vô hạn hoặc NaN
        hz_val = float(hz) if np.isfinite(hz) else 0.0
        # Tính thời gian tương ứng với frame này
        time_s = (i * hop) / sr
        frames.append(PitchFrame(time_s=time_s, f0_hz=hz_val))
    return frames


def apply_ema(values: List[float], alpha: float) -> List[float]:
    """
    Áp dụng Exponential Moving Average (EMA) để làm mịn dữ liệu
    
    EMA là phương pháp làm mịn dữ liệu theo thời gian, 
    giá trị mới có trọng số cao hơn giá trị cũ
    
    Args:
        values: Danh sách giá trị cần làm mịn
        alpha: Hệ số làm mịn (0 < alpha <= 1), alpha càng lớn thì càng nhạy với thay đổi
    
    Returns:
        Danh sách giá trị đã được làm mịn
    """
    if not values:
        return []
    smoothed: List[float] = []
    ema = values[0]  # Khởi tạo với giá trị đầu tiên
    for v in values:
        # Công thức EMA: ema_new = alpha * value + (1-alpha) * ema_old
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(float(ema))
    return smoothed


def apply_median(values: List[float], window: int) -> List[float]:
    """
    Áp dụng bộ lọc median để làm mịn dữ liệu
    
    Median filter giúp loại bỏ nhiễu và các giá trị bất thường
    bằng cách thay thế mỗi giá trị bằng median của cửa sổ xung quanh
    
    Args:
        values: Danh sách giá trị cần làm mịn
        window: Kích thước cửa sổ (phải là số lẻ)
    
    Returns:
        Danh sách giá trị đã được làm mịn
    """
    if not values or window <= 1:
        return list(values)
    
    half = window // 2
    # Thêm padding ở đầu và cuối để xử lý các phần tử biên
    padded = [values[0]] * half + list(values) + [values[-1]] * half
    out: List[float] = []
    
    for i in range(len(values)):
        # Lấy median của cửa sổ xung quanh vị trí i
        out.append(float(np.median(padded[i : i + window])))
    return out


def frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> List[float]:
    """
    Tính Root Mean Square (RMS) cho từng frame của âm thanh
    
    RMS đo năng lượng âm thanh trong mỗi frame, 
    thường được dùng để phát hiện Voice Activity Detection (VAD)
    
    Args:
        audio: Mảng âm thanh đầu vào
        frame_length: Độ dài mỗi frame
        hop_length: Khoảng cách giữa các frame
    
    Returns:
        Danh sách giá trị RMS cho từng frame
    """
    # Chuyển đổi stereo thành mono nếu cần
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    
    # Tạo cửa sổ Hanning để giảm hiệu ứng rìa (edge effects)
    window = np.hanning(frame_length).astype(np.float32)
    out: List[float] = []
    
    # Xử lý từng frame
    for start in range(0, max(0, len(audio) - frame_length + 1), hop_length):
        # Lấy frame và áp dụng cửa sổ
        fr = audio[start : start + frame_length] * window
        # Tính RMS: sqrt(mean(square(values)))
        # Thêm 1e-12 để tránh log(0) khi tính toán
        rms = float(np.sqrt(np.mean(fr**2) + 1e-12))
        out.append(rms)
    return out



