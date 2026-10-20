"""Filter known Whisper hallucinations without matching normal speech substrings."""

from pathlib import Path
import unicodedata


def _normalize(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text).casefold()
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


_PHRASES_PATH = Path(__file__).resolve().parent.parent / "hallucination_phrases.txt"
_HALLUCINATION_PHRASES = frozenset(
    normalized
    for line in _PHRASES_PATH.read_text(encoding="utf-8-sig").splitlines()
    if (normalized := _normalize(line))
)


def filter_hallucination(text: str) -> str:
    """Discard a segment only when its complete text matches a listed phrase."""
    return "" if _normalize(text) in _HALLUCINATION_PHRASES else text.strip()
