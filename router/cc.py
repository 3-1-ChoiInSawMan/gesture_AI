import os
import torch
import torch.nn as nn
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from util.labels import LABEL2IDX
from util.bigruClassifier import BiGRUClassifier
from util.config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES

idx2label = {v: k for k, v in LABEL2IDX.items()}
router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("weights/best_bigru.pt", map_location=device) 

if isinstance(checkpoint, nn.Module):
    model = checkpoint
elif isinstance(checkpoint, dict):
    model = BiGRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    raise TypeError("지원하지 않는 모델 파일 형식입니다.")

model.to(device)
model.eval()


@router.websocket("/cc")
async def jamak(websocket: WebSocket):
    await websocket.accept()
    buffer = []

    try:
        while True:
            data = await websocket.receive_json()
            sequence = np.array(data["keypoints"], dtype=np.float32)   # (60, 88)
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(x)
                pred = torch.argmax(out, dim=1).item()
                label = idx2label[pred]
            await websocket.send_text(label)

    except WebSocketDisconnect:
        raise WebSocketDisconnect("WebSocket disconnected")