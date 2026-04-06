import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import OpenAI

router = APIRouter(prefix='/ai')
Client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

WINDOW_SIZE = 60

@router.websocket('/cc')
async def jamak(websocket: WebSocket):
    await websocket.accept()
    buffer = []
    try:
        while True:
            chunk = await websocket.receive_bytes()
            buffer.append(chunk)

            if len(buffer) >= WINDOW_SIZE:
                    await websocket.send_text()
    except WebSocketDisconnect:
        pass
