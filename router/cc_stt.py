import asyncio
import io
import wave

from util.ssl_config import configure_system_truststore
configure_system_truststore()

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

from util.loadLogger import logger
logger.info("cc_stt router 로딩됨")

router = APIRouter()
model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16",
    num_workers=1
    )

@dataclass
class STTSessionState:
    audio_chunks: deque[bytes] = field(default_factory=lambda: deque(maxlen=STT_WINDOW_SIZE))
    previous_text: str = ""
    previous_tokens: list[str] = field(default_factory=list)
    committed_tokens: list[str] = field(default_factory=list)
    committed_prefix_length: int = 0
    audio_version: int = 0
    utterance_epoch: int = 0
    emitted_text: str = ""


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


def _append_text(base_text: str, addition: str) -> str:
    if not addition:
        return base_text
    if not base_text:
        return addition.strip()
    return f"{base_text.rstrip()} {addition.strip()}".strip()


def _transcribe_audio(audio_bytes: bytes) -> str:
    audio = io.BytesIO(audio_bytes)
    segments, _ = model.transcribe(
        audio,
        beam_size=1,
        language=STT_LANGUAGE,
        vad_filter=True,
    )
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
    state.emitted_text = ""
    state.audio_version += 1
    state.utterance_epoch += 1


def _consume_finalized_text(state: STTSessionState) -> str:
    finalized_text = " ".join(_finalize_tokens(state)).strip()
    text_to_emit = _find_incremental_text(state.emitted_text, finalized_text)
    if text_to_emit:
        state.emitted_text = _append_text(state.emitted_text, text_to_emit)
    return text_to_emit


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
                state.emitted_text = _append_text(state.emitted_text, text_to_emit)
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
    call_room_idx = ws.query_params.get("callRoomIdx")
    logger.info(
        "CC STT websocket accepted session=%s callRoomIdx=%s silence_timeout=%.2fs",
        session_id,
        call_room_idx,
        SILENCE_TIMEOUT_SECONDS,
    )
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
                    text_to_emit = _consume_finalized_text(state)
                    _reset_utterance_state(state)
                if text_to_emit:
                    try:
                        await ws.send_text(text_to_emit)
                    except (WebSocketDisconnect, RuntimeError):
                        return
                continue

            if not data:
                continue

            async with state_lock:
                state.audio_chunks.append(data)
                state.audio_version += 1
    except WebSocketDisconnect:
        async with state_lock:
            text_to_emit = _consume_finalized_text(state)
            _reset_utterance_state(state)
        if text_to_emit:
            logger.info(
                "CC STT finalized text dropped after disconnect session=%s text=%r",
                session_id,
                text_to_emit,
            )

    finally:
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass
        except (WebSocketDisconnect, RuntimeError):
            pass
