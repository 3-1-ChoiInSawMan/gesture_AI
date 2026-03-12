"""
수어 데이터셋 일괄 변환 스크립트 (02~16 폴더 구조 대응)

폴더 구조:
  01번: keypoint/NIA_SL_SENxxxx_REAL01_F/
  02번~: keypoint/02/F_SEN_xxxx/

사용법:
  python batch_convert.py 02        ← 02번만 변환
  python batch_convert.py 02 05     ← 02~05번 변환
  python batch_convert.py all       ← 02~16 전부 변환
  python batch_convert.py merge     ← labels.csv 합치기
"""

import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import glob

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = r"/data"
MORPHEME_BASE_DIR = r"/data/mor"
OUTPUT_NPY_DIR = r"/data/npy_output/keypoints"
OUTPUT_LABEL_DIR = r"/data/npy_output/labels"
MERGED_LABEL_CSV = r"/data/npy_output/labels.csv"
LABEL_MAP_PATH = r"/data/npy_output/label_map.json"

KEYPOINT_PARTS = [
    ("pose_keypoints_2d", 25),
    ("hand_left_keypoints_2d", 21),
    ("hand_right_keypoints_2d", 21),
    ("face_keypoints_2d", 70),
]


def extract_keypoints_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", {})
    frame_keypoints = []

    for part_name, num_points in KEYPOINT_PARTS:
        raw = people.get(part_name, [])
        if len(raw) >= num_points * 3:
            points = np.array(raw[: num_points * 3]).reshape(num_points, 3)
        else:
            points = np.zeros((num_points, 3))
        frame_keypoints.append(points)

    return np.concatenate(frame_keypoints, axis=0)


def process_single_video(args):
    folder_path, output_name, output_dir = args

    try:
        json_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith("_keypoints.json")]
        )

        if len(json_files) == 0:
            return None

        frames = []
        for jf in json_files:
            kp = extract_keypoints_from_json(os.path.join(folder_path, jf))
            frames.append(kp)

        video_keypoints = np.stack(frames, axis=0).astype(np.float32)

        output_path = os.path.join(output_dir, f"{output_name}.npy")
        np.save(output_path, video_keypoints)

        return {
            "output_name": output_name,
            "num_frames": len(json_files),
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] {output_name}: {e}")
        traceback.print_exc()
        return None


def find_keypoint_folders(keypoint_dir, folder_num):
    """
    폴더 구조에 맞게 Front 앵글 폴더 탐색

    01번: keypoint/NIA_SL_SENxxxx_REAL01_F/
    02번~: keypoint/02/F_SEN_xxxx/
    """
    tasks = []

    if folder_num == "01":
        all_folders = sorted(os.listdir(keypoint_dir))
        for f in all_folders:
            full_path = os.path.join(keypoint_dir, f)
            if os.path.isdir(full_path) and f.endswith("_F"):
                tasks.append((full_path, f))
    else:
        # 02번~: /data/02/F_SEN_xxxx/
        sub_dir = keypoint_dir
        if not os.path.exists(sub_dir):
            print(f"  [ERROR] 폴더 없음: {sub_dir}")
            return tasks

        all_folders = sorted(os.listdir(sub_dir))
        for f in all_folders:
            full_path = os.path.join(sub_dir, f)
            if not os.path.isdir(full_path):
                continue

            # F_ 로 시작하는 Front 앵글 폴더만
            if not f.startswith("F_"):
                continue

            # JSON 파일명에서 영상 ID 추출
            json_files = sorted(
                [jf for jf in os.listdir(full_path) if jf.endswith("_keypoints.json")]
            )
            if json_files:
                # NIA_SL_SEN0001_REAL02_F_000000000000_keypoints.json
                # → NIA_SL_SEN0001_REAL02_F
                sample = json_files[0]
                parts = sample.replace("_keypoints.json", "")
                name_parts = parts.rsplit("_", 1)
                if len(name_parts) == 2 and name_parts[1].isdigit():
                    output_name = name_parts[0]
                else:
                    output_name = parts
                tasks.append((full_path, output_name))

    return tasks


def convert_one_folder(folder_num):
    label_dir = os.path.join(BASE_DIR, folder_num)

    if not os.path.exists(label_dir):
        print(f"[ERROR] 폴더 없음: {label_dir}")
        return False

    keypoint_dir = label_dir
    morpheme_dir = os.path.join(MORPHEME_BASE_DIR, folder_num)

    if not os.path.exists(keypoint_dir):
        print(f"[ERROR] keypoint 폴더 없음: {keypoint_dir}")
        return False

    print(f"\n{'='*60}")
    print(f"  {folder_num}번 변환 시작")
    print(f"  경로: {label_dir}")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

    folder_tasks = find_keypoint_folders(keypoint_dir, folder_num)
    print(f"  Front 폴더 수: {len(folder_tasks)}")

    if len(folder_tasks) == 0:
        print(f"  [SKIP] Front 폴더 없음")
        return False

    # npy 변환
    tasks = [
        (full_path, output_name, OUTPUT_NPY_DIR)
        for full_path, output_name in folder_tasks
    ]

    num_workers = max(1, cpu_count() - 2)
    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_video, tasks),
            total=len(tasks),
            desc=f"{folder_num}번 npy 변환",
        ):
            if result is not None:
                results.append(result)

    print(f"  npy 변환: {len(results)}/{len(folder_tasks)}개 성공")

    # morpheme 라벨 CSV
    if os.path.exists(morpheme_dir):
        os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
        label_csv_path = os.path.join(OUTPUT_LABEL_DIR, f"labels_{folder_num}.csv")

        morpheme_files = sorted(
            [f for f in os.listdir(morpheme_dir) if f.endswith("_morpheme.json")]
        )

        rows = []
        for mf in morpheme_files:
            base_name = mf.replace("_morpheme.json", "")
            if not base_name.endswith("_F"):
                continue

            json_path = os.path.join(morpheme_dir, mf)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                morphemes = []
                for segment in data.get("data", []):
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    attributes = segment.get("attributes", [])
                    labels = [attr.get("name", "") for attr in attributes]
                    label_str = " ".join(labels)
                    morphemes.append({"start": start, "end": end, "label": label_str})

                sentence_label = " | ".join([m["label"] for m in morphemes])
                rows.append({
                    "video_id": base_name,
                    "num_morphemes": len(morphemes),
                    "sentence_label": sentence_label,
                    "morpheme_detail": json.dumps(morphemes, ensure_ascii=False),
                })
            except Exception as e:
                print(f"  [ERROR] morpheme {mf}: {e}")

        with open(label_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["video_id", "num_morphemes", "sentence_label", "morpheme_detail"]
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"  라벨 CSV: {label_csv_path} ({len(rows)}개)")
    else:
        print(f"  [WARN] morpheme 폴더 없음: {morpheme_dir}")

    return True


def merge_labels():
    print(f"\n{'='*60}")
    print("  라벨 CSV 합치기")
    print(f"{'='*60}")

    all_rows = []

    original_label = MERGED_LABEL_CSV
    backup = original_label + ".bak"
    if os.path.exists(original_label) and not os.path.exists(backup):
        import shutil
        shutil.copy2(original_label, backup)
        print(f"  기존 labels.csv 백업: {backup}")

    if os.path.exists(backup):
        with open(backup, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows_01 = list(reader)
            all_rows.extend(rows_01)
            print(f"  labels.csv.bak (01번): {len(rows_01)}개")

    label_files = sorted(glob.glob(os.path.join(OUTPUT_LABEL_DIR, "labels_*.csv")))
    for lf in label_files:
        with open(lf, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            all_rows.extend(rows)
            print(f"  {os.path.basename(lf)}: {len(rows)}개")

    seen = set()
    unique_rows = []
    for row in all_rows:
        if row["video_id"] not in seen:
            seen.add(row["video_id"])
            unique_rows.append(row)

    with open(MERGED_LABEL_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "num_morphemes", "sentence_label", "morpheme_detail"]
        )
        writer.writeheader()
        writer.writerows(unique_rows)

    print(f"\n  합친 CSV: {MERGED_LABEL_CSV}")
    print(f"  총 영상 수: {len(unique_rows)}")

    all_morphemes = set()
    for row in unique_rows:
        detail = json.loads(row["morpheme_detail"])
        for m in detail:
            all_morphemes.add(m["label"])

    sorted_morphemes = sorted(all_morphemes)
    label_map = {"<PAD>": 0, "<UNK>": 1}
    for i, morph in enumerate(sorted_morphemes):
        label_map[morph] = i + 2

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"  고유 형태소 수: {len(sorted_morphemes)}")
    print(f"  전체 클래스 수: {len(label_map)}")
    print(f"  라벨 맵: {LABEL_MAP_PATH}")


def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python batch_convert.py 02        ← 02번만")
        print("  python batch_convert.py 02 05     ← 02~05번")
        print("  python batch_convert.py all       ← 02~16 전부")
        print("  python batch_convert.py merge     ← labels.csv 합치기")
        return

    arg = sys.argv[1]

    if arg == "merge":
        merge_labels()
        return

    if arg == "all":
        start, end = 2, 16
    elif len(sys.argv) >= 3:
        start, end = int(sys.argv[1]), int(sys.argv[2])
    else:
        start = end = int(sys.argv[1])

    for num in range(start, end + 1):
        folder_num = f"{num:02d}"
        convert_one_folder(folder_num)

    print(f"\n모든 변환 완료. 'python batch_convert.py merge'로 라벨 합쳐라.")


if __name__ == "__main__":
    main()