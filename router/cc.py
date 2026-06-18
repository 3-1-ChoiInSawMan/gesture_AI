import asyncio
import json
from collections import Counter, deque
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from util.bigruClassifier import BiGRUClassifier
from util.config import (
    CC_CONF_THRESHOLD,
    CC_HANDS_DOWN_MARGIN,
    CC_HANDS_DOWN_MIN_FRAMES,
    CC_HANDS_DOWN_RATIO,
    CC_MIN_VALID_FRAMES,
    CC_NO_GESTURE_MIN_FRAMES,
    CC_PRED_EVERY_N_FRAMES,
    CC_SILENCE_TIMEOUT_SECONDS,
    CC_SMOOTHING_WINDOW,
    CC_TOP_K,
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_LAYERS,
    WINDOW_SIZE,
    LABEL2IDX  
)
from util.service.cc_service import generate_sentence_from_words, store_final_sentence
from util.loadLogger import logger

logger.info("cc router 로딩됨")

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

def _parse_word_candidates(text: str) -> list[list[str]]:
    """
    텍스트 형식을 파싱하여 word_candidates 형식으로 변환
    입력 형식: "아빠_아버지 좋다_나쁘지않다 있다_존재하다_가지고있다"
    출력 형식: [["아빠", "아버지"], ["좋다", "나쁘지않다"], ["있다", "존재하다", "가지고있다"]]
    """
    if not text or not text.strip():
        return []
    
    candidates = []
    for word_group in text.strip().split():
        if word_group:
            word_list = word_group.split("_")
            candidates.append(word_list)
    
    return candidates


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


def _are_hands_lowered(frame_vec: np.ndarray | None) -> bool:
    if frame_vec is None or frame_vec.shape != (INPUT_SIZE,):
        return False

    if not np.isfinite(frame_vec).all():
        return False

    points = frame_vec.reshape(44, 2)
    left_hand = points[:21]
    right_hand = points[21:42]
    shoulders = points[42:44]

    if np.linalg.norm(shoulders[0] - shoulders[1]) < 0.1:
        return False

    shoulder_y = max(shoulders[0][1], shoulders[1][1])
    lowered_y = shoulder_y + CC_HANDS_DOWN_MARGIN
    left_lowered_ratio = np.mean(left_hand[:, 1] > lowered_y)
    right_lowered_ratio = np.mean(right_hand[:, 1] > lowered_y)

    return (
        left_lowered_ratio >= CC_HANDS_DOWN_RATIO
        and right_lowered_ratio >= CC_HANDS_DOWN_RATIO
    )


def _majority_vote(items: deque[int]) -> int | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def _predict_sequence(sequence: list[np.ndarray]) -> list[tuple[int, float]]:
    x = np.asarray(sequence, dtype=np.float32)
    x = _normalize_sequence(x)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_k = min(CC_TOP_K, probs.numel())
    confidences, predictions = torch.topk(probs, k=top_k)
    return [
        (prediction.item(), confidence.item())
        for prediction, confidence in zip(predictions, confidences)
    ]


async def _send_debug(websocket: WebSocket, enabled: bool, event: str, **payload) -> None:
    if not enabled:
        return

    await websocket.send_json(
        {
            "type": "debug",
            "event": event,
            **payload,
        }
    )


def _extract_keypoints_payload(data: dict) -> object | None:
    keypoints = data.get("keypoints")
    if keypoints is None:
        keypoints = data.get("frame")

    if isinstance(keypoints, str):
        try:
            keypoints = json.loads(keypoints)
        except json.JSONDecodeError:
            return keypoints

    if isinstance(keypoints, dict):
        return _extract_keypoints_payload(keypoints)

    return keypoints


def _consume_frame(
    frame_vec: np.ndarray,
    *,
    seq_buffer: deque[np.ndarray],
    valid_flag_buffer: deque[bool],
    pred_history: deque[int],
    last_valid_framevec: np.ndarray,
    frame_count: int,
) -> tuple[list[str] | None, np.ndarray, dict | None]:
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
        debug = {
            "frame_count": frame_count,
            "valid": valid,
            "buffer_len": len(seq_buffer),
            "valid_frames": valid_frames,
            "can_predict": can_predict,
        }
        if len(seq_buffer) < WINDOW_SIZE:
            debug["reason"] = "buffering"
        elif valid_frames < CC_MIN_VALID_FRAMES:
            debug["reason"] = "not_enough_valid_frames"
        else:
            debug["reason"] = "prediction_interval_skipped"
        return None, last_valid_framevec, debug

    predictions = _predict_sequence(list(seq_buffer))
    if not predictions:
        return None, last_valid_framevec, None

    prediction, confidence = predictions[0]
    accepted = confidence >= CC_CONF_THRESHOLD
    pred_history.append(prediction if accepted else -1)

    candidate = idx2label.get(prediction)
    candidate_words = [
        word
        for word in (idx2label.get(prediction) for prediction, _ in predictions)
        if word
    ]

    voted = _majority_vote(pred_history)
    voted_word = idx2label.get(voted) if voted is not None and voted != -1 else None
    debug = {
        "frame_count": frame_count,
        "valid_frames": valid_frames,
        "candidate": candidate,
        "candidates": candidate_words,
        "confidence": round(confidence, 4),
        "accepted": accepted,
        "voted": voted_word,
    }

    if not voted_word:
        return None, last_valid_framevec, debug

    word_candidates = [voted_word] + [
        word for word in candidate_words if word != voted_word
    ]
    return word_candidates, last_valid_framevec, debug


async def _flush_words(
    websocket: WebSocket,
    session_id: str,
    words: list[list[str]],
    *,
    emit: bool,
    call_room_idx: object | None = None,
) -> list[list[str]]:
    if not words:
        return []

    finalized_words = list(words)
    logger.info("Flushing %s CC words for session=%s", len(finalized_words), session_id)
    sentence = await asyncio.to_thread(generate_sentence_from_words, finalized_words)
    await asyncio.to_thread(store_final_sentence, session_id, sentence, finalized_words)

    if emit and sentence:
        await websocket.send_json(
            {
                "type": "sentence",
                "sentence": sentence,
                "text": sentence,
                "callRoomIdx": call_room_idx,
            }
        )
    logger.info("문장 반환됨")
    return []


@router.websocket("/cc")
async def jamak(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or str(uuid4())
    debug_enabled = websocket.query_params.get("debug") == "1"
    ignore_hands_down = websocket.query_params.get("ignore_hands_down") == "1"
    words: list[list[str]] = []
    seq_buffer: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    valid_flag_buffer: deque[bool] = deque(maxlen=WINDOW_SIZE)
    pred_history: deque[int] = deque(maxlen=CC_SMOOTHING_WINDOW)
    last_valid_framevec = np.zeros((INPUT_SIZE,), dtype=np.float32)
    frame_count = 0
    hands_down_count = 0
    no_gesture_count = 0
    call_room_idx = websocket.query_params.get("callRoomIdx")
    logger.info(
        "CC websocket accepted session=%s silence_timeout=%.2fs",
        session_id,
        CC_SILENCE_TIMEOUT_SECONDS,
    )

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=CC_SILENCE_TIMEOUT_SECONDS,
                )
                if debug_enabled:
                    logger.debug("CC websocket frame received session=%s", session_id)
            except asyncio.TimeoutError:
                if debug_enabled:
                    logger.debug(
                        "CC silence timeout session=%s words=%s timeout=%.2fs",
                        session_id,
                        len(words),
                        CC_SILENCE_TIMEOUT_SECONDS,
                    )
                words = await _flush_words(
                    websocket,
                    session_id,
                    words,
                    emit=True,
                    call_room_idx=call_room_idx,
                )
                
                continue

            if not isinstance(data, dict):
                await _send_debug(
                    websocket,
                    debug_enabled,
                    "ignored",
                    reason="message_must_be_json_object",
                )
                continue

            if "callRoomIdx" in data:
                call_room_idx = data.get("callRoomIdx")

            keypoints = _extract_keypoints_payload(data)
            if keypoints is None:
                await _send_debug(
                    websocket,
                    debug_enabled,
                    "ignored",
                    reason="missing_keypoints_or_frame",
                )
                continue

            try:
                payload = np.asarray(keypoints, dtype=np.float32)
            except (TypeError, ValueError):
                await _send_debug(
                    websocket,
                    debug_enabled,
                    "ignored",
                    reason="keypoints_must_be_numeric",
                )
                continue

            frames: list[np.ndarray]
            if payload.shape == (INPUT_SIZE,):
                frames = [payload]
            elif payload.ndim == 2 and payload.shape[1] == INPUT_SIZE:
                frames = list(payload)
            else:
                await _send_debug(
                    websocket,
                    debug_enabled,
                    "ignored",
                    reason="invalid_keypoints_shape",
                    shape=list(payload.shape),
                    expected=[[INPUT_SIZE], ["N", INPUT_SIZE]],
                )
                continue

            for frame_vec in frames:
                valid_frame = _is_valid_frame(frame_vec)
                if valid_frame:
                    no_gesture_count = 0
                else:
                    no_gesture_count += 1
                    should_flush_no_gesture = (
                        no_gesture_count >= CC_NO_GESTURE_MIN_FRAMES
                        and bool(words)
                    )
                    await _send_debug(
                        websocket,
                        debug_enabled,
                        "no_gesture",
                        no_gesture_count=no_gesture_count,
                        will_flush=should_flush_no_gesture,
                    )
                    if should_flush_no_gesture:
                        words = await _flush_words(
                            websocket,
                            session_id,
                            words,
                            emit=True,
                            call_room_idx=call_room_idx,
                        )
                        seq_buffer.clear()
                        valid_flag_buffer.clear()
                        pred_history.clear()
                        hands_down_count = 0
                        continue

                hands_lowered = (
                    not ignore_hands_down and _are_hands_lowered(frame_vec)
                )
                if hands_lowered:
                    hands_down_count += 1
                    should_flush = (
                        hands_down_count >= CC_HANDS_DOWN_MIN_FRAMES
                        and bool(words)
                    )
                    await _send_debug(
                        websocket,
                        debug_enabled,
                        "hands_down",
                        hands_down_count=hands_down_count,
                        will_flush=should_flush,
                    )
                    if should_flush:
                        words = await _flush_words(
                            websocket,
                            session_id,
                            words,
                            emit=True,
                            call_room_idx=call_room_idx,
                        )
                        seq_buffer.clear()
                        valid_flag_buffer.clear()
                        pred_history.clear()
                        continue

                if not hands_lowered:
                    hands_down_count = 0
                frame_count += 1
                word_candidates, last_valid_framevec, debug = _consume_frame(
                    frame_vec,
                    seq_buffer=seq_buffer,
                    valid_flag_buffer=valid_flag_buffer,
                    pred_history=pred_history,
                    last_valid_framevec=last_valid_framevec,
                    frame_count=frame_count,
                )
                if debug:
                    await _send_debug(websocket, debug_enabled, "prediction", **debug)
                if debug_enabled:
                    logger.debug("CC frame processed session=%s frame=%s", session_id, frame_count)
                if word_candidates and (
                    not words or words[-1][0] != word_candidates[0]
                ):
                    # 기존에는 top3 후보까지 전달했으나, 혼잡함을 줄이기 위해
                    # 가장 높은 유사도 단어 하나만 저장하고 전송합니다.
                    # words.append(word_candidates)
                    words.append([word_candidates[0]])
                    # top_words = [candidates[0] for candidates in words if candidates]
                    await websocket.send_json(
                        {
                            "type": "word",
                            "word": word_candidates[0],
                            # "words": top_words,  # top3 후보 목록 대신 제거
                            # "word_candidates": words,  # 후보 리스트 전체 전달 주석 처리
                            "callRoomIdx": call_room_idx,
                        }
                    )
                    logger.info("words에 추가")

    except WebSocketDisconnect:
        await _flush_words(
            websocket,
            session_id,
            words,
            emit=False,
            call_room_idx=call_room_idx,
        )
        logger.info("소켓끊김")

from schema.subtitleSchema import GenerateSentenceRequest, GenerateSentenceResponse
@router.post("/cc/sentence", response_model=GenerateSentenceResponse)
async def generate_sentence(request: GenerateSentenceRequest) -> GenerateSentenceResponse:
    """
    텍스트를 받아 word_candidates로 파싱한 후 올라마를 통해 문장 생성
    
    요청 예시:
    {
        "text": "아빠_아버지 좋다_나쁘지않다 있다_존재하다_가지고있다",
        "callRoomIdx": 46
    }
    """
    logger.info("Generate sentence request received: text=%s, callRoomIdx=%s", request.text, request.callRoomIdx)
    
    # 텍스트 파싱
    word_candidates = _parse_word_candidates(request.text)
    logger.info("Parsed word_candidates: %s", word_candidates)
    
    if not word_candidates:
        logger.warning("No word candidates parsed from text")
        return GenerateSentenceResponse(
            sentence="",
            callRoomIdx=request.callRoomIdx
        )
    
    # 문장 생성
    try:
        sentence = await asyncio.to_thread(generate_sentence_from_words, word_candidates)
        logger.info("Generated sentence: %s", sentence)
    except Exception as exc:
        logger.error("Error generating sentence: %s", exc)
        return GenerateSentenceResponse(
            sentence="",
            callRoomIdx=request.callRoomIdx
        )
    
    return GenerateSentenceResponse(
        sentence=sentence,
        callRoomIdx=request.callRoomIdx
    )
