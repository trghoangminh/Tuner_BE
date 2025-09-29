from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from settings import settings


NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_to_freq(note_name: str, a4_hz: float | None = None) -> float:
    base_a4 = a4_hz or settings.a4_hz
    name = note_name.strip().upper()
    # Parse like A4, C#3, Gb2
    i = 0
    while i < len(name) and (name[i].isalpha() or name[i] in {"#", "B"}):
        i += 1
    pitch = name[:i]
    octave = int(name[i:]) if i < len(name) else 4

    # Normalize flats to sharps
    pitch = pitch.replace("DB", "C#").replace("EB", "D#").replace("GB", "F#").replace("AB", "G#").replace("BB", "A#")
    if pitch not in NOTE_NAMES_SHARP:
        raise ValueError(f"Invalid note name: {note_name}")

    semitone_index = NOTE_NAMES_SHARP.index(pitch)
    midi_number = (octave + 1) * 12 + semitone_index
    a4_midi = 69
    return base_a4 * (2 ** ((midi_number - a4_midi) / 12))


def freq_to_note(frequency_hz: float, a4_hz: float | None = None) -> Tuple[str, float]:
    if frequency_hz <= 0:
        return ("", float("nan"))
    base_a4 = a4_hz or settings.a4_hz
    a4_midi = 69
    midi = 12 * math.log2(frequency_hz / base_a4) + a4_midi
    nearest = int(round(midi))
    cents = (midi - nearest) * 100
    name = NOTE_NAMES_SHARP[nearest % 12]
    octave = nearest // 12 - 1
    return (f"{name}{octave}", cents)


def cents_between(f1: float, f2: float) -> float:
    if f1 <= 0 or f2 <= 0:
        return float("nan")
    return 1200.0 * math.log2(f2 / f1)


def nearest_target(frequency_hz: float, string_set: List[str]) -> Tuple[str, float, float]:
    if frequency_hz <= 0:
        return ("", float("nan"), float("nan"))
    distances = []
    for note_name in string_set:
        f_target = note_to_freq(note_name)
        cents = cents_between(frequency_hz, f_target)
        distances.append((abs(cents), note_name, f_target, cents))
    distances.sort(key=lambda x: x[0])
    _, name, f_target, signed_cents = distances[0]
    return name, f_target, signed_cents


@dataclass(frozen=True)
class InstrumentPreset:
    name: str
    strings: List[str]


PRESETS: Dict[str, InstrumentPreset] = {
    "guitar_standard": InstrumentPreset(
        name="Guitar Standard",
        strings=["E2", "A2", "D3", "G3", "B3", "E4"],
    ),
    "violin_standard": InstrumentPreset(
        name="Violin Standard",
        strings=["G3", "D4", "A4", "E5"],
    ),
    "ukulele_standard": InstrumentPreset(
        name="Ukulele Standard",
        strings=["G4", "C4", "E4", "A4"],
    ),
    "viola_standard": InstrumentPreset(
        name="Viola Standard",
        strings=["C3", "G3", "D4", "A4"],
    ),
    "cello_standard": InstrumentPreset(
        name="Cello Standard",
        strings=["C2", "G2", "D3", "A3"],
    ),
    "bass_standard": InstrumentPreset(
        name="Bass Standard",
        strings=["E1", "A1", "D2", "G2"],
    ),
}


