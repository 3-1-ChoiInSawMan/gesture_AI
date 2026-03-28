import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_next_index(label_dir: str) -> int:
    files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    nums = []
    for f in files:
        name = os.path.splitext(f)[0]
        if name.isdigit():
            nums.append(int(name))
    return max(nums) + 1 if nums else 0


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
    """
    handedness 대신 x 위치 기준으로 좌/우 배치
    """
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

    # 어깨 두 점이 너무 붙어 있으면 이상
    if np.linalg.norm(shoulders[0] - shoulders[1]) < 0.1:
        return False

    left_uniq = len(np.unique(np.round(left, 5), axis=0))
    right_uniq = len(np.unique(np.round(right, 5), axis=0))

    # 두 손 중 하나라도 살아있어야 함
    if max(left_uniq, right_uniq) < 5:
        return False

    return True


def is_valid_sequence(sequence, min_valid_ratio=0.9):
    if len(sequence) == 0:
        return False

    valid = sum(is_valid_frame(f) for f in sequence)
    return valid >= int(len(sequence) * min_valid_ratio)


def draw_text(img, text, y=30, color=(0, 255, 0)):
    cv2.putText(
        img,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def collect_data(
    label: str,
    save_dir: str = "dataset",
    frames_per_sample: int = 60,
    max_attempt_frames: int = 300,
):
    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    sample_idx = get_next_index(label_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    print(f"[INFO] label = {label}")
    print("[INFO] SPACE: 녹화 시작")
    print("[INFO] Q    : 종료")

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

            # 추출은 원본
            input_frame = frame.copy()
            # 보기는 좌우반전
            view_frame = cv2.flip(frame, 1)

            draw_text(
                view_frame,
                f"label={label} next={sample_idx:03d} | SPACE: record | Q: quit"
            )

            cv2.imshow("collect", view_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            import time

            if key == ord(" "):
                for _ in range(100):
                    time.sleep(1)

                    current_idx = sample_idx
                    print(f"[INFO] recording sample {current_idx:03d} ...")

                    sequence = []
                    attempts = 0

                    while len(sequence) < frames_per_sample and attempts < max_attempt_frames:
                        attempts += 1

                        ret, frame = cap.read()
                        if not ret:
                            continue

                        input_frame = frame.copy()
                        show_frame = cv2.flip(frame, 1)

                        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                        results_hands = hands_model.process(rgb)
                        results_pose = pose_model.process(rgb)

                        shoulders = extract_shoulders(results_pose)
                        hands = extract_hands(results_hands)

                        if shoulders is None or len(hands) == 0:
                            draw_text(
                                show_frame,
                                f"SKIP | shoulders={shoulders is not None} hands={len(hands)}",
                                color=(0, 0, 255),
                            )
                            cv2.imshow("collect", show_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                cap.release()
                                cv2.destroyAllWindows()
                                return
                            continue

                        left_hand, right_hand, hand_count = assign_left_right(hands, shoulders)
                        frame_vec = normalize_frame(left_hand, right_hand, shoulders)

                        if not is_valid_frame(frame_vec):
                            draw_text(
                                show_frame,
                                f"INVALID | hands={hand_count}",
                                color=(0, 0, 255),
                            )
                            cv2.imshow("collect", show_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                cap.release()
                                cv2.destroyAllWindows()
                                return
                            continue

                        sequence.append(frame_vec)

                        if results_hands.multi_hand_landmarks:
                            for hand_landmarks in results_hands.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(
                                    show_frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS
                                )

                        if results_pose.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                show_frame,
                                results_pose.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS
                            )

                        draw_text(
                            show_frame,
                            f"REC {label} {current_idx:03d} | valid {len(sequence)}/{frames_per_sample} | attempt {attempts}/{max_attempt_frames}",
                            color=(0, 255, 255),
                        )

                        cv2.imshow("collect", show_frame)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                    if len(sequence) != frames_per_sample:
                        print("[WARN] 유효 프레임 부족 -> 저장 안 함")
                        continue

                    sequence = np.array(sequence, dtype=np.float32)

                    if not is_valid_sequence(sequence, min_valid_ratio=0.9):
                        print("[WARN] 시퀀스 품질 낮음 -> 저장 안 함")
                        continue

                    save_path = os.path.join(label_dir, f"{current_idx:03d}.npy")
                    np.save(save_path, sequence)

                    print(f"[SAVED] {save_path} | shape={sequence.shape}")
                    sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    label = input("수집할 라벨명 입력: ").strip()
    if not label:
        raise ValueError("라벨명이 비어 있음")

    collect_data(
        label=label,
        save_dir="dataset",
        frames_per_sample=60,
        max_attempt_frames=300,
    )