import io
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import torch
from util.config import STT_WINDOW_SIZE

router = APIRouter()
model = WhisperModel("small", device = 'cuda' if torch.cuda.is_available() else 'cpu')

@router.websocket('/cc_stt')
async def stt_cc(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            audio = io.BytesIO(data)
            segments, info = model.transcribe(audio, beam_size=5)
            for segment in segments:
                await ws.send_text(segment.text)
    except WebSocketDisconnect:
        raise WebSocketDisconnect("WebSocket disconnected")