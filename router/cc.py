import asyncio
import json
import os
from collections import Counter, deque
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from util.bigruClassifier import BiGRUClassifier, GRUClassifier
from util.config import (
    CC_HANDS_DOWN_MARGIN,
    CC_HANDS_DOWN_MIN_FRAMES,
    CC_HANDS_DOWN_RATIO,
    CC_MIN_VALID_FRAMES,
    CC_NO_GESTURE_MIN_FRAMES,
    CC_NONE_MIN_PREDICTIONS,
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
from util.service.cc_service import store_final_sentence
from util.loadLogger import logger
from util.gesture_prediction import is_none_label, select_word_candidates
from util.gesture_preprocessing import prepare_sequence
from util.coordinate_logging import log_received_coordinates

logger.info("cc router 로딩됨")

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = Path(__file__).resolve().parent.parent / "weights"
configured_model_path = os.getenv("CC_MODEL_PATH")
model_path = Path(configured_model_path) if configured_model_path else model_dir / "best_gru.pt"
if not configured_model_path and not model_path.exists():
    for filename in ("best_bigru.pt", "best_bigru30.pt"):
        fallback_model_path = model_dir / filename
        if fallback_model_path.exists():
            logger.warning("Using fallback CC model path: %s", fallback_model_path)
            model_path = fallback_model_path
            break

checkpoint = torch.load(model_path, map_location=device)
sequence_normalization = (
    checkpoint.get("sequence_normalization", True)
    if isinstance(checkpoint, dict) else True
)
logger.info("CC model=%s sequence_normalization=%s", model_path, sequence_normalization)

if isinstance(checkpoint, dict) and "idx2label" in checkpoint:
    idx2label = checkpoint["idx2label"]
    if idx2label and isinstance(next(iter(idx2label.keys())), str):
        idx2label = {int(key): value for key, value in idx2label.items()}
else:
    idx2label = {v: k for k, v in LABEL2IDX.items()}

if not any(is_none_label(label) for label in idx2label.values()):
    logger.warning(
        "CC model has no none class. Visible idle hands require a model "
        "retrained with none samples; confidence filtering alone is insufficient."
    )

if isinstance(checkpoint, nn.Module):
    model = checkpoint
elif isinstance(checkpoint, dict):
    model = GRUClassifier(  # GRU 테스트용, 나중에 BiGRU로 바꾸면 됨
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
        # Check before splitting: no_gesture is a single none alias.
        if word_group and not is_none_label(word_group):
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


def _predict_sequence(sequence: list[np.ndarray]) -> list[tuple[int, float]]:
    x = np.asarray(sequence, dtype=np.float32)
    x = prepare_sequence(x, sequence_normalization=sequence_normalization)
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
        # Before the first valid observation, invalid input is padded with zeros.
        # Replace that startup padding with real coordinates before inference;
        # zero shoulders cannot be normalized. Flags remain False so padding
        # never counts toward the required number of observed valid frames.
        if not _is_valid_frame(last_valid_framevec):
            startup_count = len(seq_buffer)
            seq_buffer.clear()
            seq_buffer.extend(frame_vec.copy() for _ in range(startup_count))
        last_valid_framevec = frame_vec.copy()
        seq_buffer.append(frame_vec)
        valid_flag_buffer.append(True)
    else:
        seq_buffer.append(last_valid_framevec.copy())
        valid_flag_buffer.append(False)
        pred_history.clear()
        return None, last_valid_framevec, {
            "frame_count": frame_count,
            "valid": False,
            "reason": "invalid_frame",
        }

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
    word_candidates, debug = select_word_candidates(predictions, idx2label, pred_history)
    debug.update(frame_count=frame_count, valid_frames=valid_frames)
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

    # Each entry is one accepted prediction within a single hands-up segment.
    votes = Counter(candidate[0] for candidate in words if candidate)
    if not votes:
        return []
    word, support = votes.most_common(1)[0]
    total = sum(votes.values())
    if support < 3 or support / total < 0.60:
        logger.info("Discarding ambiguous gesture session=%s votes=%s", session_id, dict(votes))
        return []
    finalized_words = [[word]]
    logger.info("Flushing %s CC words for session=%s", len(finalized_words), session_id)
    # Sentence rewriting is disabled: display the recognized word verbatim.
    sentence = word
    await asyncio.to_thread(store_final_sentence, session_id, sentence, finalized_words)

    if not sentence:
        logger.warning("CC sentence generation returned empty session=%s", session_id)
        return []

    if emit and sentence:
        try:
            await websocket.send_json(
                {
                    "type": "word",
                    "word": word,
                    "text": sentence,
                    "callRoomIdx": call_room_idx,
                }
            )
        except (WebSocketDisconnect, RuntimeError):
            logger.warning(
                "CC sentence generated but not delivered session=%s sentence=%r",
                session_id,
                sentence,
            )
            raise
        logger.info("CC sentence sent session=%s sentence=%r", session_id, sentence)
    else:
        logger.info(
            "CC sentence generated without websocket emit session=%s sentence=%r",
            session_id,
            sentence,
        )
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
    received_frame_count = 0
    hands_down_count = 0
    no_gesture_count = 0
    none_prediction_count = 0
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
                seq_buffer.clear()
                valid_flag_buffer.clear()
                pred_history.clear()
                last_valid_framevec = np.zeros((INPUT_SIZE,), dtype=np.float32)
                hands_down_count = 0
                no_gesture_count = 0
                none_prediction_count = 0
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
                logger.warning("CC coordinates rejected session=%s reason=non_numeric", session_id)
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
                logger.warning("CC coordinates rejected session=%s shape=%s expected=(88,) or (N,88)", session_id, payload.shape)
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
                received_frame_count += 1
                if received_frame_count == 1 or received_frame_count % 30 == 0:
                    log_received_coordinates(logger, session_id, call_room_idx,
                                             received_frame_count, frame_vec, valid_frame)
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
                    # Lowered hands are a boundary, never recognition input.
                    # Clear even when there are no words to flush.
                    seq_buffer.clear()
                    valid_flag_buffer.clear()
                    pred_history.clear()
                    none_prediction_count = 0
                    last_valid_framevec = np.zeros((INPUT_SIZE,), dtype=np.float32)
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
                    if debug.get("reason") == "none_label":
                        none_prediction_count += 1
                    elif debug.get("reason") != "prediction_interval_skipped":
                        none_prediction_count = 0
                # None predictions do not split a hands-up gesture. Only a
                # confirmed hands-down/tracking-loss boundary or timeout ends it.
                if debug_enabled:
                    logger.debug("CC frame processed session=%s frame=%s", session_id, frame_count)
                if word_candidates:
                    words.append([word_candidates[0]])

    except (WebSocketDisconnect, RuntimeError):
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
    텍스트에서 인식 단어를 추출하여 문장화 없이 그대로 반환
    
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
    
    # Keep the endpoint response compatible, without calling an LLM.
    sentence = " ".join(candidates[0] for candidates in word_candidates if candidates)

    return GenerateSentenceResponse(
        sentence=sentence,
        callRoomIdx=request.callRoomIdx
    )
