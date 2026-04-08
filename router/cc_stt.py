import io
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

WINDOW_SIZE = 15 

@router.websocket('/cc_stt')
async def stt_cc():
    pass