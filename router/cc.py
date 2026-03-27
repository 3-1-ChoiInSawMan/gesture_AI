import io
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix='/ai')

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
                    await websocket.send_text()
    except WebSocketDisconnect:
        pass
