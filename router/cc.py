import io
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

router = APIRouter(prefix='/ai')
model = WhisperModel("base", device="cuda", compute_type='float16')

WINDOW_SIZE = 15 

@router.websocket('/cc')
async def jamak(websocket: WebSocket):
    await websocket.accept()
    buffer = []
    try:
        while True:
            chunk = await websocket.receive_bytes()
            buffer.append(chunk)

            if len(buffer) >= WINDOW_SIZE:
                audio_data = b"".join(buffer)
                buffer = []

                segments, _ = model.transcribe(io.BytesIO(audio_data))
                text = " ".join([seg.text for seg in segments]).strip()

                if text:
                    await websocket.send_text(text)
    except WebSocketDisconnect:
        pass
