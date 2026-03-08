from util.npyConvertUtil.keypoints import extract_keypoints_from_json
import os
import traceback
import numpy as np

def process_single_video(args):
    folder_path, folder_name, output_dir = args

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

        output_path = os.path.join(output_dir, f"{folder_name}.npy")
        np.save(output_path, video_keypoints)

        return {
            "folder_name": folder_name,
            "num_frames": len(json_files),
            "shape": video_keypoints.shape,
        }

    except Exception as e:
        print(f"[ERROR] {folder_name}: {e}")
        traceback.print_exc()
        return None