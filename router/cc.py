import os
import torch
import torch.nn as nn
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from util.labels import LABEL2IDX

idx2label = {v: k for k, v in LABEL2IDX.items()}

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE = 60
INPUT_SIZE = 88       # 한 프레임 keypoints 길이에 맞게 수정
HIDDEN_SIZE = 128     # 학습할 때 쓴 값으로 수정
NUM_LAYERS = 2        # 학습할 때 쓴 값으로 수정
NUM_CLASSES = 29      # 클래스 개수에 맞게 수정


class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, num_layers=2, num_classes=29, dropout=0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 60, 88)
        out, _ = self.gru(x)
        last = out[:, -1, :]   # (B, hidden_dim*2)
        return self.head(last)


checkpoint = torch.load("best_bigru.pt", map_location=device)

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