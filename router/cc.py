import asyncio
from collections import Counter, deque
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from util.bigruClassifier import BiGRUClassifier
from util.config import (
    CC_CONF_THRESHOLD,
    CC_MIN_VALID_FRAMES,
    CC_PRED_EVERY_N_FRAMES,
    CC_SMOOTHING_WINDOW,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_LAYERS,
    SILENCE_TIMEOUT_SECONDS,
    WINDOW_SIZE,
    LABEL2IDX  
)
from util.service.cc_service import generate_sentence_from_words, store_final_sentence

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = Path(__file__).resolve().parent.parent / "weights" / "best_bigru.pt"
checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, dict) and "idx2label" in checkpoint:
    idx2label = checkpoint["idx2label"]
    if idx2label and isinstance(next(iter(idx2label.keys())), str):
        idx2label = {int(key): value for key, value in idx2label.items()}
else:
    idx2label = {v: k for k, v in LABEL2IDX.items()}

if isinstance(checkpoint, nn.Module):
    model = checkpoint
elif isinstance(checkpoint, dict):
    model = BiGRUClassifier(
        checkpoint.get("input_dim", INPUT_SIZE),
        HIDDEN_SIZE,
        NUM_LAYERS,
        len(idx2label) or NUM_CLASSES,
    )
    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state_dict)
else:
    raise TypeError("지원하지 않는 모델 파일 형식입니다.")

model.to(device)
model.eval()


def _normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True) + 1e-6
    return (sequence - mean) / std


def _is_valid_frame(frame_vec: np.ndarray | None) -> bool:
    if frame_vec is None or frame_vec.shape != (INPUT_SIZE,):
        return False

    if not np.isfinite(frame_vec).all():
        return False

    points = frame_vec.reshape(44, 2)
    unique_points = np.unique(np.round(points, 5), axis=0)
    if len(unique_points) < 10:
        return False

    left_hand = points[:21]
    right_hand = points[21:42]
    shoulders = points[42:44]

    if np.linalg.norm(shoulders[0] - shoulders[1]) < 0.1:
        return False

    left_unique = len(np.unique(np.round(left_hand, 5), axis=0))
    right_unique = len(np.unique(np.round(right_hand, 5), axis=0))
    if max(left_unique, right_unique) < 5:
        return False

    return True


def _majority_vote(items: deque[int]) -> int | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def _predict_sequence(sequence: list[np.ndarray]) -> tuple[int, float]:
    x = np.asarray(sequence, dtype=np.float32)
    x = _normalize_sequence(x)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    prediction = torch.argmax(probs).item()
    confidence = probs[prediction].item()
    return prediction, confidence


def _consume_frame(
    frame_vec: np.ndarray,
    *,
    seq_buffer: deque[np.ndarray],
    valid_flag_buffer: deque[bool],
    pred_history: deque[int],
    last_valid_framevec: np.ndarray,
    frame_count: int,
) -> tuple[str | None, np.ndarray]:
    valid = _is_valid_frame(frame_vec)

    if valid:
        last_valid_framevec = frame_vec.copy()
        seq_buffer.append(frame_vec)
        valid_flag_buffer.append(True)
    else:
        seq_buffer.append(last_valid_framevec.copy())
        valid_flag_buffer.append(False)

    valid_frames = sum(valid_flag_buffer)
    can_predict = len(seq_buffer) == WINDOW_SIZE and valid_frames >= CC_MIN_VALID_FRAMES
    if not can_predict or frame_count % CC_PRED_EVERY_N_FRAMES != 0:
        return None, last_valid_framevec

    prediction, confidence = _predict_sequence(list(seq_buffer))
    pred_history.append(prediction if confidence >= CC_CONF_THRESHOLD else -1)

    voted = _majority_vote(pred_history)
    if voted is None or voted == -1:
        return None, last_valid_framevec

    return idx2label.get(voted), last_valid_framevec


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
    seq_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    valid_flag_buffer: deque[bool] = deque(maxlen=WINDOW_SIZE)
    pred_history: deque[int] = deque(maxlen=CC_SMOOTHING_WINDOW)
    last_valid_framevec = np.zeros((INPUT_SIZE,), dtype=np.float32)
    frame_count = 0

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
                payload = np.asarray(keypoints, dtype=np.float32)
            except (TypeError, ValueError):
                continue

            if payload.shape == (INPUT_SIZE,):
                frame_count += 1
                word, last_valid_framevec = _consume_frame(
                    payload,
                    seq_buffer=seq_buffer,
                    valid_flag_buffer=valid_flag_buffer,
                    pred_history=pred_history,
                    last_valid_framevec=last_valid_framevec,
                    frame_count=frame_count,
                )
            elif payload.shape == (WINDOW_SIZE, INPUT_SIZE):
                word = None
                for frame_vec in payload:
                    frame_count += 1
                    word, last_valid_framevec = _consume_frame(
                        frame_vec,
                        seq_buffer=seq_buffer,
                        valid_flag_buffer=valid_flag_buffer,
                        pred_history=pred_history,
                        last_valid_framevec=last_valid_framevec,
                        frame_count=frame_count,
                    )
            else:
                continue

            if word and (not words or words[-1] != word):
                words.append(word)

    except WebSocketDisconnect:
        await _flush_words(
            websocket,
            session_id,
            words,
            emit=False,
        )
