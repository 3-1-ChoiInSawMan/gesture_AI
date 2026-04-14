from typing import Any

from util.mongo_connect import col
from util.config import SPEAKER_KEYS, TEXT_KEYS, LIST_KEYS


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _extract_speaker(doc: dict[str, Any]) -> str:
    for key in SPEAKER_KEYS:
        value = _stringify_value(doc.get(key))
        if value:
            return value
    return ""


def _extract_text(doc: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    for key in TEXT_KEYS:
        value = _stringify_value(doc.get(key))
        if value:
            speaker = _extract_speaker(doc)
            lines.append(f"{speaker}: {value}" if speaker else value)

    for key in LIST_KEYS:
        value = doc.get(key)
        if not isinstance(value, list):
            continue

        for item in value:
            if isinstance(item, dict):
                lines.extend(_extract_text(item))
            else:
                text = _stringify_value(item)
                if text:
                    lines.append(text)

    return lines


def load_message(session_id: str | None = None, limit: int = 100) -> list[str]:
    query: dict[str, Any] = {}
    if session_id:
        query["session_id"] = session_id

    cursor = col.find(query).sort("_id", 1).limit(limit)

    messages: list[str] = []
    for doc in cursor:
        messages.extend(_extract_text(doc))

    seen: set[str] = set()
    normalized_messages: list[str] = []
    for message in messages:
        cleaned = " ".join(message.split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized_messages.append(cleaned)

    return normalized_messages
