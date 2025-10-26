import math
import re

from typing_extensions import ParamSpec

P = ParamSpec("P")
NOTE_RE = re.compile(r"^(?P<note>[A-Ga-g])(?P<accidental>[#♯𝄪b!♭𝄫♮]*)(?P<octave>[+-]?\d+)?$")
KEY_IN_OCTAVE = 12


def midi2note(midi: float) -> str:
    pitch_map = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]
    midi = round(midi)
    octave = (midi // KEY_IN_OCTAVE) - 1
    pitch = pitch_map[midi % KEY_IN_OCTAVE]
    return f"{pitch}{octave}"


def note2midi(note: str) -> int:
    pitch_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    acc_map = {
        "#": 1,
        "": 0,
        "b": -1,
        "!": -1,
        "♯": 1,
        "𝄪": 2,
        "♭": -1,
        "𝄫": -2,
        "♮": 0,
    }

    if (match := NOTE_RE.match(note)) is None:
        msg = f"Invalid note format: {note!r}"
        raise ValueError(msg)

    pitch = match["note"].upper()
    offset = sum(acc_map[o] for o in match["accidental"])
    octave = match["octave"]

    octave = int(octave) if octave else 0

    note_value = KEY_IN_OCTAVE * (octave + 1) + pitch_map[pitch] + offset

    return round(note_value)


def hz2midi(hz: float, a4_midi: int = 69, base_freq: float = 440.0) -> float:
    return a4_midi + KEY_IN_OCTAVE * math.log2(hz / base_freq)


def midi2hz(midi: float, a4_midi: int = 69, base_freq: float = 440.0) -> float:
    return base_freq * 2 ** ((midi - a4_midi) / KEY_IN_OCTAVE)