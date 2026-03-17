import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_keypoints_2d(person: dict) -> np.ndarray:
    """
    JSON people 객체에서 2D 키포인트 추출
    반환: (137, 3) - [x, y, confidence]
    """
    def parse_kp(flat_list):
        arr = np.array(flat_list, dtype=np.float32)
        return arr.reshape(-1, 3)  # (N, 3)

    face  = parse_kp(person["face_keypoints_2d"])   # (70, 3)
    pose  = parse_kp(person["pose_keypoints_2d"])   # (25, 3)
    lhand = parse_kp(person["hand_left_keypoints_2d"])  # (21, 3)
    rhand = parse_kp(person["hand_right_keypoints_2d"]) # (21, 3)

    return np.concatenate([pose, lhand, rhand, face], axis=0)  # (137, 3)


def load_video_keypoints(json_dir: str, seq_len: int = 120) -> np.ndarray:
    """
    한 영상 폴더의 프레임 JSON들 → (seq_len, 137, 3)
    """
    json_files = sorted(Path(json_dir).glob("*.json"))

    frames = []
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        people = data.get("people")

        # people 없거나 비어있으면 제로 프레임
        if not people:
            frames.append(np.zeros((137, 3), dtype=np.float32))
            continue

        # dict면 바로, list면 첫번째 사람만
        person = people if isinstance(people, dict) else people[0]
        frames.append(extract_keypoints_2d(person))

    frames = np.array(frames, dtype=np.float32)  # (T, 137, 3)
    return pad_or_trim(frames, seq_len)


def pad_or_trim(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """프레임 수를 seq_len으로 통일"""
    T = len(seq)
    if T >= seq_len:
        return seq[:seq_len]
    pad = np.tile(seq[-1:], (seq_len - T, 1, 1))
    return np.concatenate([seq, pad], axis=0)


def preprocess_dataset(root_dir: str, output_dir: str, seq_len: int = 120):
    """
    root_dir 구조:
        root_dir/
            NIA_SL_WORD0001_REAL01_F/
                NIA_SL_WORD0001_REAL01_F_000000000000_keypoints.json
                NIA_SL_WORD0001_REAL01_F_000000000001_keypoints.json
                ...
            NIA_SL_WORD0002_REAL01_F/
                ...

    output_dir에 {video_id}.npy 저장
    """
    os.makedirs(output_dir, exist_ok=True)

    video_dirs = sorted([
        d for d in Path(root_dir).iterdir() if d.is_dir()
    ])

    failed = []

    for video_dir in tqdm(video_dirs, desc="전처리 중"):
        video_id = video_dir.name
        out_path = Path(output_dir) / f"{video_id}.npy"

        if out_path.exists():  # 이미 처리된 거 스킵
            continue

        try:
            kp = load_video_keypoints(str(video_dir), seq_len)
            np.save(out_path, kp)
            # kp = load_video_keypoints(
            #     "/home/ingyu/004.수어영상/1.Training/라벨링데이터/REAL/WORD/03/NIA_SL_WORD0001_REAL03_F",
            #     seq_len=120
            #     )   
            # print(kp.shape)
        except Exception as e:
            print(f"[실패] {video_id}: {e}")
            failed.append(video_id)

    print(f"\n완료: {len(video_dirs) - len(failed)}개")
    print(f"실패: {len(failed)}개")
    if failed:
        with open(Path(output_dir) / "failed.txt", "w") as f:
            f.write("\n".join(failed))


if __name__ == "__main__":
    preprocess_dataset(
        root_dir="/home/ingyu/004.수어영상/1.Training/라벨링데이터/REAL/WORD/13",
        output_dir="/home/ingyu/004.수어영상/preprocessed/word_npy",
        seq_len=120
    )