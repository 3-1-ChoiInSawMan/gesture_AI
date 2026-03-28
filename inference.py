import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque, Counter

# =========================
# 설정
# =========================
CKPT_PATH = "best_bigru.pt"
SEQ_LEN = 60
INPUT_DIM = 88
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRED_EVERY_N_FRAMES = 2      # 너무 자주 추론하면 흔들림
CONF_THRESHOLD = 0.70        # 이보다 낮으면 unsure
SMOOTHING_WINDOW = 7         # 최근 예측 7개 다수결
MIN_VALID_FRAMES = 45        # 60프레임 중 최소 유효 프레임 수
CAM_INDEX = 0

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# =========================
# 모델
# =========================
class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.3):
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
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


# =========================
# 수집 코드와 같은 전처리
# =========================
def extract_shoulders(results_pose):
    if not results_pose.pose_landmarks:
        return None

    l = results_pose.pose_landmarks.landmark[11]
    r = results_pose.pose_landmarks.landmark[12]

    shoulders = np.array([
        [l.x, l.y],
        [r.x, r.y],
    ], dtype=np.float32)

    if not np.isfinite(shoulders).all():
        return None

    dist = np.linalg.norm(shoulders[0] - shoulders[1])
    if dist < 1e-6:
        return None

    return shoulders


def extract_hands(results_hands):
    hands = []

    if not results_hands.multi_hand_landmarks:
        return hands

    for hand_landmarks in results_hands.multi_hand_landmarks:
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y])

        coords = np.array(coords, dtype=np.float32)

        if coords.shape == (21, 2) and np.isfinite(coords).all():
            hands.append(coords)

    return hands


def assign_left_right(hands, shoulders):
    left = np.zeros((21, 2), dtype=np.float32)
    right = np.zeros((21, 2), dtype=np.float32)

    if len(hands) == 0:
        return left, right, 0

    if len(hands) == 1:
        hand = hands[0]
        hand_cx = hand[:, 0].mean()
        body_cx = shoulders[:, 0].mean()

        if hand_cx < body_cx:
            left = hand
        else:
            right = hand
        return left, right, 1

    hands = sorted(hands, key=lambda h: h[:, 0].mean())
    left = hands[0]
    right = hands[1]
    return left, right, 2


def normalize_frame(left_hand, right_hand, shoulders):
    center = (shoulders[0] + shoulders[1]) / 2.0
    shoulder_dist = np.linalg.norm(shoulders[0] - shoulders[1])

    if shoulder_dist < 1e-6:
        return None

    left_hand = (left_hand - center) / shoulder_dist
    right_hand = (right_hand - center) / shoulder_dist
    shoulders = (shoulders - center) / shoulder_dist

    frame_vec = np.concatenate([
        left_hand.flatten(),   # 42
        right_hand.flatten(),  # 42
        shoulders.flatten()    # 4
    ], axis=0).astype(np.float32)

    return frame_vec  # (88,)


def is_valid_frame(frame_vec):
    if frame_vec is None:
        return False

    if frame_vec.shape != (88,):
        return False

    if not np.isfinite(frame_vec).all():
        return False

    pts = frame_vec.reshape(44, 2)

    uniq_all = np.unique(np.round(pts, 5), axis=0)
    if len(uniq_all) < 10:
        return False

    left = pts[:21]
    right = pts[21:42]
    shoulders = pts[42:44]

    if np.linalg.norm(shoulders[0] - shoulders[1]) < 0.1:
        return False

    left_uniq = len(np.unique(np.round(left, 5), axis=0))
    right_uniq = len(np.unique(np.round(right, 5), axis=0))

    if max(left_uniq, right_uniq) < 5:
        return False

    return True


def normalize_sequence(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


# =========================
# 모델 로드 / 추론
# =========================
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    idx2label = ckpt["idx2label"]
    if isinstance(list(idx2label.keys())[0], str):
        idx2label = {int(k): v for k, v in idx2label.items()}

    model = BiGRUClassifier(
        input_dim=ckpt.get("input_dim", INPUT_DIM),
        hidden_dim=128,
        num_layers=2,
        num_classes=len(idx2label),
        dropout=0.3,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, idx2label


def predict_sequence(model, seq_60):
    x = np.asarray(seq_60, dtype=np.float32)   # (60, 88)
    x = normalize_sequence(x)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(3, probs.shape[0])
    values, indices = torch.topk(probs, k=topk)

    pred = indices[0].item()
    conf = values[0].item()

    return pred, conf, indices.cpu().numpy().tolist(), values.cpu().numpy().tolist()


# =========================
# 유틸
# =========================
def draw_text(img, text, y=30, color=(0, 255, 0), scale=0.7):
    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def majority_vote(items):
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


# =========================
# 메인
# =========================
def main():
    print("DEVICE:", DEVICE)
    model, idx2label = load_model()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    seq_buffer = deque(maxlen=SEQ_LEN)
    valid_flag_buffer = deque(maxlen=SEQ_LEN)
    pred_history = deque(maxlen=SMOOTHING_WINDOW)

    last_valid_framevec = np.zeros((INPUT_DIM,), dtype=np.float32)

    stable_label = "waiting..."
    stable_conf = 0.0
    top3_text = []
    frame_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands_model, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose_model:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 프레임 읽기 실패")
                break

            frame_count += 1

            # 보기용만 좌우반전
            view = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_hands = hands_model.process(rgb)
            results_pose = pose_model.process(rgb)

            shoulders = extract_shoulders(results_pose)
            hands = extract_hands(results_hands)

            frame_vec = None
            hand_count = len(hands)

            if shoulders is not None and len(hands) > 0:
                left_hand, right_hand, _ = assign_left_right(hands, shoulders)
                frame_vec = normalize_frame(left_hand, right_hand, shoulders)

            valid = is_valid_frame(frame_vec)

            # 실시간은 프레임이 비면 버퍼가 깨지니까
            # invalid면 마지막 유효 프레임으로 채우고 valid_flag는 False로 기록
            if valid:
                last_valid_framevec = frame_vec.copy()
                seq_buffer.append(frame_vec)
                valid_flag_buffer.append(True)
            else:
                seq_buffer.append(last_valid_framevec.copy())
                valid_flag_buffer.append(False)

            # 랜드마크 그리기
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        view,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    view,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # 충분히 쌓였을 때만 예측
            valid_frames = sum(valid_flag_buffer)
            can_predict = (len(seq_buffer) == SEQ_LEN and valid_frames >= MIN_VALID_FRAMES)

            if can_predict and frame_count % PRED_EVERY_N_FRAMES == 0:
                pred_idx, conf, top_ids, top_scores = predict_sequence(model, list(seq_buffer))

                if conf >= CONF_THRESHOLD:
                    pred_history.append(pred_idx)
                else:
                    pred_history.append(-1)

                voted = majority_vote(pred_history)

                if voted is None or voted == -1:
                    stable_label = "unsure"
                    stable_conf = conf
                else:
                    stable_label = idx2label[voted]
                    stable_conf = conf

                top3_text = [
                    f"1. {idx2label[top_ids[0]]}: {top_scores[0]:.3f}",
                    f"2. {idx2label[top_ids[1]]}: {top_scores[1]:.3f}" if len(top_ids) > 1 else "",
                    f"3. {idx2label[top_ids[2]]}: {top_scores[2]:.3f}" if len(top_ids) > 2 else "",
                ]
                top3_text = [x for x in top3_text if x]

            # UI
            color = (0, 255, 0) if stable_label not in ["unsure", "waiting..."] else (0, 255, 255)

            draw_text(view, f"pred: {stable_label}", 35, color=color, scale=1.0)
            draw_text(view, f"conf: {stable_conf:.3f}", 70, color=color)
            draw_text(view, f"buffer: {len(seq_buffer)}/{SEQ_LEN}", 105)
            draw_text(view, f"valid_frames: {valid_frames}/{SEQ_LEN}", 140)
            draw_text(view, f"hands: {hand_count}", 175)
            draw_text(view, f"frame_valid: {valid}", 210, color=(0, 255, 0) if valid else (0, 0, 255))
            draw_text(view, "Q: quit | C: clear buffer", 245, color=(255, 255, 0))

            y = 290
            for line in top3_text:
                draw_text(view, line, y, color=(255, 255, 255))
                y += 35

            # 진행 바
            bar_x1, bar_y1 = 10, 660
            bar_x2, bar_y2 = 610, 700
            cv2.rectangle(view, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 2)
            fill_w = int((len(seq_buffer) / SEQ_LEN) * (bar_x2 - bar_x1))
            cv2.rectangle(view, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 255, 0), -1)

            cv2.imshow("realtime gesture inference", view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                seq_buffer.clear()
                valid_flag_buffer.clear()
                pred_history.clear()
                stable_label = "waiting..."
                stable_conf = 0.0
                top3_text = []
                print("[INFO] buffer cleared")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()