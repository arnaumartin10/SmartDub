"""Map ARPAbet phonemes to Preston Blair visemes and frame timelines."""

from __future__ import annotations

# Canonical CMU ARPAbet inventory (stress digits are stripped before lookup).
STANDARD_ARPABET_PHONEMES: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW", "B", "CH", "D", "DH", "F", "G", "HH",
    "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH",
    "V", "W", "Y", "Z", "ZH",
)

# Preston Blair groups plus explicit articulatory categories for the remaining
# English consonants; REST is reserved for silence and pause tokens.
PRESTON_BLAIR_VISEMES: dict[str, tuple[str, ...]] = {
    "AI": ("AA", "AE", "AH", "AW", "AY", "AX"),
    "E": ("EH", "ER", "EY", "IH", "IY", "IX"),
    "O": ("AO", "OW", "OY"),
    "U": ("UH", "UW", "UX"),
    "FV": ("F", "V"),
    "L": ("L",),
    "MBP": ("B", "M", "P"),
    "WQ": ("W", "Q"),
    "CHJSH": ("CH", "JH", "SH", "ZH"),
    "TH": ("DH", "TH"),
    "SZ": ("S", "Z"),
    "TDN": ("T", "D", "N", "DX"),
    "KGHNG": ("K", "G", "NG"),
    "R": ("R",),
    "YHH": ("Y", "HH"),
    "REST": ("", "SIL", "SP", "SILENCE", "PAUSE"),
}

_PHONEME_TO_VISEME = {
    phoneme: viseme for viseme, phonemes in PRESTON_BLAIR_VISEMES.items() for phoneme in phonemes
}


def _normalise_phoneme(value: str) -> str:
    token = value.strip().upper()
    # ARPAbet stress digits (AH0, IY1, etc.) do not change the mouth shape.
    return token.rstrip("012")


def phonemes_to_visemes(aligned_phonemes: list[dict]) -> list[dict]:
    """Map aligned ARPAbet phoneme records while preserving timing fields."""
    sequence = []
    for item in aligned_phonemes:
        phoneme = str(item.get("phoneme", ""))
        normalised = _normalise_phoneme(phoneme)
        try:
            viseme = _PHONEME_TO_VISEME[normalised]
        except KeyError as exc:
            raise ValueError(
                f"No viseme mapping for ARPAbet phoneme {phoneme!r} "
                f"(normalised as {normalised!r})"
            ) from exc
        mapped = dict(item)
        mapped["viseme"] = viseme
        sequence.append(mapped)
    return sequence


def build_viseme_timeline(viseme_sequence: list[dict], total_frames: int) -> list[str]:
    """Return one viseme label per frame, defaulting gaps to ``REST``."""
    if total_frames < 0:
        raise ValueError(f"total_frames must be non-negative, got {total_frames}")
    timeline = ["REST"] * total_frames
    for item in viseme_sequence:
        viseme = str(item.get("viseme", "REST"))
        start = max(0, int(item.get("start_frame", 0)))
        end = min(total_frames - 1, int(item.get("end_frame", -1)))
        if end >= start:
            timeline[start : end + 1] = [viseme] * (end - start + 1)
    return timeline
