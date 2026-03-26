import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_hand_keypoints(results_hands):
    """
    항상 같은 길이로 반환:
    - left hand: 21 * 2
    - right hand: 21 * 2
    총 84차원
    """
    left_hand = np.zeros((21, 2), dtype=np.float32)
    right_hand = np.zeros((21, 2), dtype=np.float32)

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hand_landmarks, handedness in zip(
            results_hands.multi_hand_landmarks,
            results_hands.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" or "Right"

            coords = []
            for lm in hand_landmarks.landmark:
                coords.append([lm.x, lm.y])
            coords = np.array(coords, dtype=np.float32)

            # mediapipe 기준 handedness 사용
            if label == "Left":
                left_hand = coords
            elif label == "Right":
                right_hand = coords

    return np.concatenate([
        left_hand.flatten(),
        right_hand.flatten()
    ], axis=0)  # (84,)


def extract_pose_keypoints(results_pose):
    """
    포즈는 어깨 2개만 사용
    left shoulder = 11, right shoulder = 12
    총 4차원
    """
    pose_vec = np.zeros((2, 2), dtype=np.float32)

    if results_pose.pose_landmarks:
        l_shoulder = results_pose.pose_landmarks.landmark[11]
        r_shoulder = results_pose.pose_landmarks.landmark[12]

        pose_vec[0] = [l_shoulder.x, l_shoulder.y]
        pose_vec[1] = [r_shoulder.x, r_shoulder.y]

    return pose_vec.flatten()  # (4,)


def normalize_frame_keypoints(frame_vec):
    """
    frame_vec:
    [left_hand(42), right_hand(42), shoulders(4)] = 총 88차원

    정규화:
    - 어깨 중앙을 원점으로 이동
    - 어깨 거리로 스케일 정규화
    """
    frame_vec = frame_vec.copy().astype(np.float32)

    hands = frame_vec[:84].reshape(42, 2)
    shoulders = frame_vec[84:].reshape(2, 2)

    left_sh = shoulders[0]
    right_sh = shoulders[1]

    center = (left_sh + right_sh) / 2.0
    shoulder_dist = np.linalg.norm(left_sh - right_sh)

    if shoulder_dist < 1e-6:
        shoulder_dist = 1.0

    hands = (hands - center) / shoulder_dist
    shoulders = (shoulders - center) / shoulder_dist

    return np.concatenate([hands.flatten(), shoulders.flatten()], axis=0)


def extract_keypoints(results_hands, results_pose):
    hand_vec = extract_hand_keypoints(results_hands)   # (84,)
    pose_vec = extract_pose_keypoints(results_pose)    # (4,)
    frame_vec = np.concatenate([hand_vec, pose_vec], axis=0)  # (88,)
    frame_vec = normalize_frame_keypoints(frame_vec)
    return frame_vec.astype(np.float32)


def get_next_index(label_dir):
    existing = [f for f in os.listdir(label_dir) if f.endswith(".npy")]
    if not existing:
        return 0

    nums = []
    for f in existing:
        name = os.path.splitext(f)[0]
        if name.isdigit():
            nums.append(int(name))

    if not nums:
        return 0

    return max(nums) + 1


def collect_data(label, save_dir="dataset", frames_per_sample=30):
    label_dir = os.path.join(save_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    sample_idx = get_next_index(label_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    print(f"\n[INFO] label = {label}")
    print("[INFO] space: 녹화 시작")
    print("[INFO] q    : 종료\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 프레임을 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)
            view = frame.copy()

            cv2.putText(
                view,
                f"label={label} next={sample_idx:03d} | SPACE: record | Q: quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("collect", view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                print(f"[INFO] recording sample {sample_idx:03d} ...")

                sequence = []

                for frame_no in range(frames_per_sample):
                    ret, frame = cap.read()
                    if not ret:
                        print("[ERROR] 녹화 중 프레임 읽기 실패")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    results_hands = hands.process(rgb)
                    results_pose = pose.process(rgb)

                    keypoints = extract_keypoints(results_hands, results_pose)
                    sequence.append(keypoints)

                    draw_frame = frame.copy()

                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                draw_frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS
                            )

                    if results_pose.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            draw_frame,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )

                    cv2.putText(
                        draw_frame,
                        f"REC {label} {sample_idx:03d} frame {frame_no + 1}/{frames_per_sample}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    cv2.imshow("collect", draw_frame)

                    key2 = cv2.waitKey(1) & 0xFF
                    if key2 == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                sequence = np.array(sequence, dtype=np.float32)

                if len(sequence) == frames_per_sample:
                    save_path = os.path.join(label_dir, f"{sample_idx:03d}.npy")
                    np.save(save_path, sequence)
                    print(f"[SAVED] {save_path} | shape={sequence.shape}")
                    sample_idx += 1
                else:
                    print("[WARN] 녹화 길이가 부족해서 저장하지 않았습니다.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    label = input("수집할 라벨명을 입력하세요: ").strip()
    if not label:
        raise ValueError("라벨명이 비어 있습니다.")

    collect_data(
        label=label,
        save_dir="dataset",
        frames_per_sample=60
    )