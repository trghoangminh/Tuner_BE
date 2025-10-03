# Import các thư viện cần thiết cho xử lý âm nhạc
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from settings import settings

# Danh sách tên các nốt nhạc theo thang âm chromatic (12 nốt)
# Sử dụng ký hiệu sharp (#) thay vì flat (b)
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_freq(note_name: str, a4_hz: float | None = None) -> float:
    """
    Chuyển đổi tên nốt nhạc thành tần số (Hz)
    
    Ví dụ: "A4" -> 440.0 Hz, "C#3" -> 138.59 Hz
    
    Args:
        note_name: Tên nốt nhạc (ví dụ: "A4", "C#3", "Gb2")
        a4_hz: Tần số của nốt A4 (mặc định 440 Hz)
    
    Returns:
        Tần số tương ứng với nốt nhạc (Hz)
    """
    base_a4 = a4_hz or settings.a4_hz
    name = note_name.strip().upper()
    
    # Phân tích tên nốt: tách phần pitch (C, C#, D...) và octave (4, 3, 2...)
    i = 0
    while i < len(name) and (name[i].isalpha() or name[i] in {"#", "B"}):
        i += 1
    pitch = name[:i]  # Phần tên nốt (C, C#, D...)
    octave = int(name[i:]) if i < len(name) else 4  # Phần octave (4, 3, 2...)

    # Chuyển đổi ký hiệu flat (b) thành sharp (#) để chuẩn hóa
    # Ví dụ: Db -> C#, Eb -> D#, Gb -> F#, Ab -> G#, Bb -> A#
    pitch = pitch.replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    if pitch not in NOTE_NAMES_SHARP:
        raise ValueError(f"Invalid note name: {note_name}")

    # Tính MIDI number và chuyển đổi thành tần số
    semitone_index = NOTE_NAMES_SHARP.index(pitch)  # Vị trí trong thang âm chromatic (0-11)
    midi_number = (octave + 1) * 12 + semitone_index  # MIDI number
    a4_midi = 69  # MIDI number của A4
    # Công thức: freq = A4_freq * 2^((midi_number - A4_midi) / 12)
    return base_a4 * (2 ** ((midi_number - a4_midi) / 12))


def freq_to_note(frequency_hz: float, a4_hz: float | None = None) -> Tuple[str, float]:
    """
    Chuyển đổi tần số (Hz) thành tên nốt nhạc và cents
    
    Ví dụ: 440.0 Hz -> ("A4", 0.0), 445.0 Hz -> ("A4", 19.55)
    
    Args:
        frequency_hz: Tần số cần chuyển đổi (Hz)
        a4_hz: Tần số của nốt A4 (mặc định 440 Hz)
    
    Returns:
        Tuple (tên_nốt, cents_lệch) - cents dương nghĩa là cao hơn nốt chuẩn
    """
    if frequency_hz <= 0:
        return ("", float("nan"))
    
    base_a4 = a4_hz or settings.a4_hz
    a4_midi = 69  # MIDI number của A4
    
    # Chuyển đổi tần số thành MIDI number
    midi = 12 * math.log2(frequency_hz / base_a4) + a4_midi
    nearest = int(round(midi))  # MIDI number gần nhất
    
    # Tính cents: 1 semitone = 100 cents, 1 octave = 1200 cents
    cents = (midi - nearest) * 100
    
    # Lấy tên nốt và octave từ MIDI number
    name = NOTE_NAMES_SHARP[nearest % 12]  # Tên nốt (0-11)
    octave = nearest // 12 - 1  # Octave
    return (f"{name}{octave}", cents)


def cents_between(f1: float, f2: float) -> float:
    """
    Tính khoảng cách giữa hai tần số theo đơn vị cents
    
    Cents là đơn vị đo khoảng cách âm thanh: 1200 cents = 1 octave
    
    Args:
        f1: Tần số thứ nhất (Hz)
        f2: Tần số thứ hai (Hz)
    
    Returns:
        Khoảng cách theo cents (dương nếu f2 > f1)
    """
    if f1 <= 0 or f2 <= 0:
        return float("nan")
    return 1200.0 * math.log2(f2 / f1)


def nearest_target(frequency_hz: float, string_set: List[str]) -> Tuple[str, float, float]:
    """
    Tìm nốt gần nhất trong bộ dây của nhạc cụ
    
    Hàm này giúp xác định nốt nào trong preset nhạc cụ gần với tần số phát hiện nhất
    
    Args:
        frequency_hz: Tần số phát hiện (Hz)
        string_set: Danh sách các nốt trong preset nhạc cụ
    
    Returns:
        Tuple (tên_nốt_gần_nhất, tần_số_nốt_đó, cents_lệch)
    """
    if frequency_hz <= 0:
        return ("", float("nan"), float("nan"))
    
    distances = []
    # Tính khoảng cách đến từng nốt trong preset
    for note_name in string_set:
        f_target = note_to_freq(note_name)
        cents = cents_between(frequency_hz, f_target)
        distances.append((abs(cents), note_name, f_target, cents))
    
    # Sắp xếp theo khoảng cách và lấy nốt gần nhất
    distances.sort(key=lambda x: x[0])
    _, name, f_target, signed_cents = distances[0]
    return name, f_target, signed_cents


@dataclass(frozen=True)
class InstrumentPreset:
    """Lớp lưu trữ thông tin preset của nhạc cụ"""
    name: str  # Tên hiển thị của nhạc cụ
    strings: List[str]  # Danh sách các nốt chuẩn của nhạc cụ (từ dây thấp đến cao)


# Từ điển chứa các preset nhạc cụ phổ biến
# Mỗi preset định nghĩa các nốt chuẩn mà nhạc cụ đó có thể phát ra
PRESETS: Dict[str, InstrumentPreset] = {
    "guitar_standard": InstrumentPreset(
        name="Guitar Standard",
        strings=["E2", "A2", "D3", "G3", "B3", "E4"],  # 6 dây guitar chuẩn
    ),
    "violin_standard": InstrumentPreset(
        name="Violin Standard", 
        strings=["G3", "D4", "A4", "E5"],  # 4 dây violin chuẩn
    ),
    "ukulele_standard": InstrumentPreset(
        name="Ukulele Standard",
        strings=["G4", "C4", "E4", "A4"],  # 4 dây ukulele chuẩn
    ),
    "viola_standard": InstrumentPreset(
        name="Viola Standard",
        strings=["C3", "G3", "D4", "A4"],  # 4 dây viola chuẩn
    ),
    "cello_standard": InstrumentPreset(
        name="Cello Standard",
        strings=["C2", "G2", "D3", "A3"],  # 4 dây cello chuẩn
    ),
    "bass_standard": InstrumentPreset(
        name="Bass Standard",
        strings=["E1", "A1", "D2", "G2"],  # 4 dây bass chuẩn
    ),
}


