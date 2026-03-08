KEYPOINT_DIR = r"D:\수어 영상\01_label\keypoint\01"
MORPHEME_DIR = r"D:\수어 영상\01_label\morpheme\morpheme\01"
OUTPUT_NPY_DIR = r"C:\npy_output\keypoints"
OUTPUT_LABEL_CSV = r"C:\npy_output\labels.csv"

ANGLE_FILTER = "_F"

KEYPOINT_PARTS = [
    ("pose_keypoints_2d", 25),
    ("hand_left_keypoints_2d", 21),
    ("hand_right_keypoints_2d", 21),
    ("face_keypoints_2d", 70),
]
TOTAL_KEYPOINTS = sum(n for _, n in KEYPOINT_PARTS)