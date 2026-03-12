KEYPOINT_DIR = r"/data/02"
MORPHEME_DIR = r"/data/mor/02"
OUTPUT_NPY_DIR = r"/data/npy_output/keypoints"
OUTPUT_LABEL_CSV = r"/data/npy_output/labels.csv"

ANGLE_FILTER = "_F"

KEYPOINT_PARTS = [
    ("pose_keypoints_2d", 25),
    ("hand_left_keypoints_2d", 21),
    ("hand_right_keypoints_2d", 21),
    ("face_keypoints_2d", 70),
]
TOTAL_KEYPOINTS = sum(n for _, n in KEYPOINT_PARTS)