from collections import deque
from datetime import UTC, datetime
import io
import asyncio
from uuid import uuid4

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

from util.config import STT_WINDOW_SIZE, SILENCE_TIMEOUT_SECONDS
from util.mongo_connect import col

router = APIRouter()
model = WhisperModel("small", device='cuda' if torch.cuda.is_available() else 'cpu')

def _extract_text(segments) -> str:
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in text.split() if token]


def _longest_common_prefix_length(previous_tokens: list[str], current_tokens: list[str]) -> int:
    common_length = 0
    for previous_token, current_token in zip(previous_tokens, current_tokens):
        if previous_token != current_token:
            break
        common_length += 1
    return common_length


def _find_incremental_text(previous_text: str, current_text: str) -> str:
    if not current_text or current_text == previous_text:
        return ""

    if current_text.startswith(previous_text):
        return current_text[len(previous_text):].strip()

    max_overlap = min(len(previous_text), len(current_text))
    for overlap in range(max_overlap, 0, -1):
        if previous_text.endswith(current_text[:overlap]):
            return current_text[overlap:].strip()

    return current_text


def _store_final_subtitle(session_id: str, finalized_tokens: list[str]) -> None:
    finalized_text = " ".join(finalized_tokens).strip()
    if not finalized_text:
        return

    col.insert_one(
        {
            "session_id": session_id,
            "text": finalized_text,
            "source": "stt",
            "is_final": True,
            "created_at": datetime.now(UTC),
        }
    )


@router.websocket('/cc_stt')
async def stt_cc(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("session_id") or str(uuid4())
    audio_chunks = deque(maxlen=STT_WINDOW_SIZE)
    previous_text = ""
    previous_tokens: list[str] = []
    committed_tokens: list[str] = []
    committed_prefix_length = 0

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    ws.receive_bytes(),
                    timeout=SILENCE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                remaining_tokens = previous_tokens[committed_prefix_length:]
                finalized_tokens = committed_tokens + remaining_tokens
                _store_final_subtitle(session_id, finalized_tokens)

                audio_chunks.clear()
                previous_text = ""
                previous_tokens = []
                committed_tokens = []
                committed_prefix_length = 0
                continue

            if not data:
                continue

            audio_chunks.append(data)
            audio = io.BytesIO(b"".join(audio_chunks))
            segments, _ = model.transcribe(audio, beam_size=1)
            current_text = _extract_text(segments)
            current_tokens = _tokenize(current_text)
            common_prefix_length = _longest_common_prefix_length(previous_tokens, current_tokens)

            if common_prefix_length < committed_prefix_length:
                committed_prefix_length = common_prefix_length

            if common_prefix_length > committed_prefix_length:
                committed_tokens.extend(previous_tokens[committed_prefix_length:common_prefix_length])
                committed_prefix_length = common_prefix_length

            incremental_text = _find_incremental_text(previous_text, current_text)

            if incremental_text:
                await ws.send_text(incremental_text)
                previous_text = current_text

            previous_tokens = current_tokens
    except WebSocketDisconnect:
        remaining_tokens = previous_tokens[committed_prefix_length:]
        finalized_tokens = committed_tokens + remaining_tokens
        _store_final_subtitle(session_id, finalized_tokens)
