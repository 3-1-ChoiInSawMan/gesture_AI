"""
word_npy 정규화 후처리 스크립트
기존 npy에 정규화만 적용해서 새 폴더에 저장
"""

import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

INPUT_DIR  = "/data/word_npy"        # 기존 npy 경로
OUTPUT_DIR = "/data/word_npy_norm"   # 정규화된 npy 저장 경로
WIDTH  = 1920.0
HEIGHT = 1080.0
SEQ_LEN = 120


def pad_or_trim(seq, seq_len=SEQ_LEN):
    T = len(seq)
    if T >= seq_len:
        return seq[:seq_len]
    pad = np.tile(seq[-1:], (seq_len - T, 1, 1))
    return np.concatenate([seq, pad], axis=0)


def process(filename):
    in_path  = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(out_path):
        return "skip"

    try:
        kp = np.load(in_path)           # (T, 137, 3)
        kp = pad_or_trim(kp)            # (120, 137, 3)
        kp[..., 0] /= WIDTH             # x 정규화
        kp[..., 1] /= HEIGHT            # y 정규화
        np.save(out_path, kp)
        return "ok"
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")
        return "fail"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")]
    print(f"총 {len(files)}개 처리 시작")

    workers = max(1, cpu_count() - 2)
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap_unordered(process, files), total=len(files)))

    ok   = results.count("ok")
    skip = results.count("skip")
    fail = results.count("fail")
    print(f"\n완료: {ok}개 | 스킵: {skip}개 | 실패: {fail}개")