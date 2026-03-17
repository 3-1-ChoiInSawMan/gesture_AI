from util.parseMorpheme import parse_morpheme
import os
import numpy as np
import json
import csv

def build_label_csv(morpheme_dir, angle_filter, output_csv):
    """
    morpheme JSON들을 파싱해서 라벨 매핑 CSV 생성
    """
    rows = []

    morpheme_files = sorted(
        [f for f in os.listdir(morpheme_dir) if f.endswith("_morpheme.json")]
    )

    for mf in morpheme_files:
        # Front 앵글만 필터
        # 파일명: NIA_SL_SEN0001_REAL01_F_morpheme.json
        # 앵글은 _morpheme.json 앞의 한 글자
        base_name = mf.replace("_morpheme.json", "")
        if not base_name.endswith(angle_filter):
            continue

        json_path = os.path.join(morpheme_dir, mf)
        try:
            morphemes = parse_morpheme(json_path)
            # 전체 문장 라벨 (형태소들을 순서대로 이어붙임)
            sentence_label = " | ".join([m["label"] for m in morphemes])

            rows.append(
                {
                    "video_id": base_name,
                    "num_morphemes": len(morphemes),
                    "sentence_label": sentence_label,
                    "morpheme_detail": json.dumps(morphemes, ensure_ascii=False),
                }
            )
        except Exception as e:
            print(f"[ERROR] morpheme {mf}: {e}")

    # CSV 저장
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "num_morphemes", "sentence_label", "morpheme_detail"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] 라벨 CSV 저장: {output_csv}")
    print(f"  총 {len(rows)}개 영상 라벨 처리됨")

    return rows