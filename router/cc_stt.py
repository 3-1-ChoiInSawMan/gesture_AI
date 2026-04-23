import asyncio
import io
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

from util.config import (
    SILENCE_TIMEOUT_SECONDS,
    STT_INFERENCE_INTERVAL_SECONDS,
    STT_LANGUAGE,
    STT_WINDOW_SIZE,
)
from util.mongo_connect import col

router = APIRouter()
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel("base", device=device, compute_type=compute_type)


@dataclass
class STTSessionState:
    audio_chunks: deque[bytes] = field(default_factory=lambda: deque(maxlen=STT_WINDOW_SIZE))
    previous_text: str = ""
    previous_tokens: list[str] = field(default_factory=list)
    committed_tokens: list[str] = field(default_factory=list)
    committed_prefix_length: int = 0
    audio_version: int = 0
    utterance_epoch: int = 0


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


def _transcribe_audio(audio_bytes: bytes) -> str:
    audio = io.BytesIO(audio_bytes)
    segments, _ = model.transcribe(audio, beam_size=1, language=STT_LANGUAGE)
    return _extract_text(segments)


def _finalize_tokens(state: STTSessionState) -> list[str]:
    remaining_tokens = state.previous_tokens[state.committed_prefix_length:]
    return state.committed_tokens + remaining_tokens


def _reset_utterance_state(state: STTSessionState) -> None:
    state.audio_chunks.clear()
    state.previous_text = ""
    state.previous_tokens = []
    state.committed_tokens = []
    state.committed_prefix_length = 0
    state.audio_version += 1
    state.utterance_epoch += 1


async def _run_inference_loop(
    ws: WebSocket,
    state: STTSessionState,
    state_lock: asyncio.Lock,
) -> None:
    last_processed_version = -1

    while True:
        await asyncio.sleep(STT_INFERENCE_INTERVAL_SECONDS)

        async with state_lock:
            if not state.audio_chunks or state.audio_version == last_processed_version:
                continue

            audio_bytes = b"".join(state.audio_chunks)
            snapshot_version = state.audio_version
            snapshot_epoch = state.utterance_epoch

        try:
            current_text = await asyncio.to_thread(_transcribe_audio, audio_bytes)
        except Exception:
            continue

        current_tokens = _tokenize(current_text)
        text_to_emit = ""

        async with state_lock:
            if snapshot_epoch != state.utterance_epoch:
                continue

            common_prefix_length = _longest_common_prefix_length(state.previous_tokens, current_tokens)

            if common_prefix_length < state.committed_prefix_length:
                state.committed_prefix_length = common_prefix_length

            if common_prefix_length > state.committed_prefix_length:
                state.committed_tokens.extend(
                    state.previous_tokens[state.committed_prefix_length:common_prefix_length]
                )
                state.committed_prefix_length = common_prefix_length

            incremental_text = _find_incremental_text(state.previous_text, current_text)

            if incremental_text:
                text_to_emit = incremental_text
                state.previous_text = current_text

            state.previous_tokens = current_tokens
            last_processed_version = snapshot_version

        if text_to_emit:
            try:
                await ws.send_text(text_to_emit)
            except (WebSocketDisconnect, RuntimeError):
                return


@router.websocket('/cc_stt')
async def stt_cc(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("session_id") or str(uuid4())
    state = STTSessionState()
    state_lock = asyncio.Lock()
    inference_task = asyncio.create_task(_run_inference_loop(ws, state, state_lock))

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    ws.receive_bytes(),
                    timeout=SILENCE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                async with state_lock:
                    finalized_tokens = _finalize_tokens(state)
                    _reset_utterance_state(state)

                await asyncio.to_thread(_store_final_subtitle, session_id, finalized_tokens)
                continue

            if not data:
                continue

            async with state_lock:
                state.audio_chunks.append(data)
                state.audio_version += 1
    except WebSocketDisconnect:
        async with state_lock:
            finalized_tokens = _finalize_tokens(state)
            _reset_utterance_state(state)

        await asyncio.to_thread(_store_final_subtitle, session_id, finalized_tokens)
    finally:
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass
        except (WebSocketDisconnect, RuntimeError):
            pass
