import os
import torch
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import OpenAI

router = APIRouter()
Client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("best_bigru.pt", map_location=device)
model.to(device)
model.eval()

WINDOW_SIZE = 60

@router.websocket("/cc")
async def jamak(websocket: WebSocket):
    await websocket.accept()
    buffer = []
    try:
        while True:
            data = await websocket.receive_json()
            keypoints = data["keypoints"]

            buffer.append(keypoints)

            if len(buffer) >= WINDOW_SIZE:
                sequence = np.array(buffer[-WINDOW_SIZE:], dtype=np.float32)
                x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(x)
                    pred = torch.argmax(out, dim=1).item()

                await websocket.send_text(str(pred))
                buffer = []

    except WebSocketDisconnect:
        pass