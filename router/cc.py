import asyncio
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from util.bigruClassifier import BiGRUClassifier
from util.config import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_LAYERS,
    SILENCE_TIMEOUT_SECONDS,
    WINDOW_SIZE,
)
from util.labels import LABEL2IDX
from util.service.cc_service import generate_sentence_from_words, store_final_sentence

idx2label = {v: k for k, v in LABEL2IDX.items()}
router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = Path(__file__).resolve().parent.parent / "weights" / "best_bigru.pt"
checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, nn.Module):
    model = checkpoint
elif isinstance(checkpoint, dict):
    model = BiGRUClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)
    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state_dict)
else:
    raise TypeError("지원하지 않는 모델 파일 형식입니다.")

model.to(device)
model.eval()


async def _flush_words(
    websocket: WebSocket,
    session_id: str,
    words: list[str],
    *,
    emit: bool, ) -> list[str]:
    if not words:
        return []

    finalized_words = list(words)
    sentence = await asyncio.to_thread(generate_sentence_from_words, finalized_words)
    await asyncio.to_thread(store_final_sentence, session_id, sentence, finalized_words)

    if emit and sentence:
        await websocket.send_text(sentence)

    return []


@router.websocket("/cc")
async def jamak(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or str(uuid4())
    words: list[str] = []

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=SILENCE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                words = await _flush_words(
                    websocket,
                    session_id,
                    words,
                    emit=True,
                )
                continue

            if not isinstance(data, dict):
                continue

            keypoints = data.get("keypoints")
            if keypoints is None:
                continue

            try:
                sequence = np.asarray(keypoints, dtype=np.float32)
            except (TypeError, ValueError):
                continue

            if sequence.shape != (WINDOW_SIZE, INPUT_SIZE):
                continue

            x = torch.from_numpy(sequence).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                prediction = torch.argmax(logits, dim=1).item()
                word = idx2label[prediction]

            if not words or words[-1] != word:
                words.append(word)

    except WebSocketDisconnect:
        await _flush_words(
            websocket,
            session_id,
            words,
            emit=False,
        )
