from config import KEYPOINT_PARTS
import numpy as np
import json

def extract_keypoints_from_json(json_path):
    """
    단일 프레임 JSON에서 2D 키포인트 추출
    Returns: np.array shape (137, 3) — (x, y, confidence)
    """
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