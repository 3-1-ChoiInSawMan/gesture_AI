import asyncio
import json

from util.ssl_config import configure_system_truststore
configure_system_truststore()

from uuid import uuid4

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

from util.config import (
    STT_SILENCE_TIMEOUT_SECONDS,
    STT_LANGUAGE,
    STT_MAX_UTTERANCE_BYTES,
)

from util.loadLogger import logger
from util.stt_filter import filter_hallucination
from util.stt_audio import STTAudioBuffer
logger.info("cc_stt router 로딩됨")

router = APIRouter()
model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16",
    num_workers=1
    )

def _extract_text(segments) -> str:
    return " ".join(
        text for segment in segments if (text := filter_hallucination(segment.text))
    ).strip()


def _transcribe_audio(audio) -> str:
    segments, _ = model.transcribe(
        audio,
        beam_size=1,
        language=STT_LANGUAGE,
        vad_filter=True,
    )
    return _extract_text(segments)


async def _finish_utterance(ws: WebSocket, audio_buffer: STTAudioBuffer, session_id: str) -> None:
    if not audio_buffer.pending:
        return
    try:
        audio = await asyncio.to_thread(audio_buffer.take_audio)
    except Exception:
        logger.warning(
            "CC STT audio decode incomplete/invalid session=%s bytes=%s header=%s; "
            "waiting for more audio (headerless streams require recorder restart)",
            session_id, len(audio_buffer.data), bytes(audio_buffer.data[:16]).hex(),
        )
        return
    if audio is None:
        return
    logger.info("CC STT utterance inference session=%s samples=%s", session_id, len(audio))
    try:
        text = await asyncio.to_thread(_transcribe_audio, audio)
    except Exception:
        logger.exception("CC STT utterance inference failed session=%s", session_id)
        return
    if text:
        await ws.send_text(text)


@router.websocket('/cc_stt')
async def stt_cc(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("session_id") or str(uuid4())
    logger.info(
        "CC STT websocket accepted session=%s callRoomIdx=%s silence_timeout=%.2fs",
        session_id, ws.query_params.get("callRoomIdx"), STT_SILENCE_TIMEOUT_SECONDS,
    )
    audio_buffer = STTAudioBuffer(STT_MAX_UTTERANCE_BYTES)
    loop = asyncio.get_running_loop()
    deadline = None

    try:
        while True:
            try:
                # This measures binary-audio inactivity, not acoustic silence.
                # Continuous microphone clients must signal end_of_utterance.
                timeout = max(0.0, deadline - loop.time()) if deadline is not None else None
                message = await asyncio.wait_for(ws.receive(), timeout=timeout)
            except asyncio.TimeoutError:
                await _finish_utterance(ws, audio_buffer, session_id)
                deadline = None
                continue

            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                try:
                    audio_buffer.append(data)
                except BufferError:
                    logger.warning("CC STT audio stream too large session=%s", session_id)
                    await ws.close(code=1009, reason="Restart recorder: audio stream buffer limit")
                    break
                deadline = loop.time() + STT_SILENCE_TIMEOUT_SECONDS
            elif message.get("text"):
                try:
                    event = json.loads(message["text"])
                except (TypeError, ValueError):
                    continue
                if isinstance(event, dict) and event.get("type") == "end_of_utterance":
                    await _finish_utterance(ws, audio_buffer, session_id)
                    deadline = None
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        # The peer cannot receive a result after disconnect; release pending audio.
        audio_buffer.clear()
