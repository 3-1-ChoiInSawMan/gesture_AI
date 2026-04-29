import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def downsample_sequence(seq: np.ndarray, target_len: int = 60) -> np.ndarray:
    """
    seq: (T, D)
    target_len 길이로 다운샘플
    """
    if seq.ndim != 2:
        raise ValueError(f"sequence shape must be (T, D), got {seq.shape}")

    T = len(seq)

    if T == 0:
        raise ValueError("empty sequence")

    if T == target_len:
        return seq.astype(np.float32)

    if T < target_len:
        # 짧은 시퀀스는 마지막 프레임 반복해서 패딩
        pad_count = target_len - T
        pad = np.repeat(seq[-1][None, :], pad_count, axis=0)
        return np.concatenate([seq, pad], axis=0).astype(np.float32)

    indices = np.linspace(0, T - 1, target_len).astype(int)
    return seq[indices].astype(np.float32)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    간단 정규화:
    - 전체 프레임 기준 평균 0
    - 전체 프레임 기준 표준편차 1
    """
    seq = seq.astype(np.float32)

    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6

    return (seq - mean) / std


def preprocess_sequence(seq: np.ndarray, target_len: int = 20) -> np.ndarray:
    """
    전처리 통합
    """
    seq = downsample_sequence(seq, target_len=target_len)
    seq = normalize_sequence(seq)
    return seq.astype(np.float32)


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    seq1: (T1, D)
    seq2: (T2, D)
    return: DTW distance
    """
    if seq1.ndim != 2 or seq2.ndim != 2:
        raise ValueError("both sequences must be 2D arrays of shape (T, D)")

    T1, D1 = seq1.shape
    T2, D2 = seq2.shape

    if D1 != D2:
        raise ValueError(f"feature dim mismatch: {D1} != {D2}")

    dp = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dp[i, j] = cost + min(
                dp[i - 1, j],      # insertion
                dp[i, j - 1],      # deletion
                dp[i - 1, j - 1],  # match
            )

    return float(dp[T1, T2])


def load_dataset(root_dir: str, target_len: int = 20):
    """
    dataset/
      hello/
        001.npy
        002.npy
      thanks/
        001.npy
        002.npy
    """
    X = []
    y = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"dataset root not found: {root_dir}")

    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if not class_names:
        raise ValueError("no class folders found")

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith(".npy")])

        if not npy_files:
            print(f"[WARN] no npy files in class: {class_name}")
            continue

        for fname in npy_files:
            path = os.path.join(class_dir, fname)

            try:
                seq = np.load(path)

                if seq.ndim != 2:
                    print(f"[SKIP] invalid shape {seq.shape} -> {path}")
                    continue

                seq = preprocess_sequence(seq, target_len=target_len)

                X.append(seq)
                y.append(class_name)

            except Exception as e:
                print(f"[SKIP] failed to load {path}: {e}")

    if not X:
        raise ValueError("no valid samples loaded")

    return X, y


class DTWClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train length mismatch")

        self.X_train = X_train
        self.y_train = y_train

    def predict_one(self, x):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("model is not fitted")

        best_dist = float("inf")
        best_label = None

        for train_seq, label in zip(self.X_train, self.y_train):
            dist = dtw_distance(x, train_seq)

            if dist < best_dist:
                best_dist = dist
                best_label = label

        return best_label, best_dist

    def predict(self, X_test):
        preds = []

        for x in X_test:
            label, _ = self.predict_one(x)
            preds.append(label)

        return preds


def print_dataset_stats(y):
    from collections import Counter

    counter = Counter(y)
    print("\n[Dataset Stats]")
    print(f"총 샘플 수: {len(y)}")
    print(f"클래스 수: {len(counter)}")
    for cls, count in sorted(counter.items()):
        print(f"{cls}: {count}")


import matplotlib.font_manager as fm

# 1) 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

def main():
    dataset_dir = "dataset"   # 여기를 네 폴더 경로로 수정 가능
    target_len = 20
    test_size = 0.2
    random_state = 42

    X, y = load_dataset(dataset_dir, target_len=target_len)
    print_dataset_stats(y)

    # 클래스당 샘플이 너무 적으면 stratify 에러 날 수 있음
    # 그럴 땐 test_size 줄이거나 클래스별 샘플 수를 늘려야 함
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    clf = DTWClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\n[Confusion Matrix]")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
    print("saved: confusion_matrix.png")

    print("\n[Evaluation]")
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("\nclassification_report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n[Sample Predictions]")
    for i in range(min(20, len(X_test))):
        pred, dist = clf.predict_one(X_test[i])
        print(f"GT={y_test[i]:<15} PRED={pred:<15} DTW_DIST={dist:.4f}")

if __name__ == "__main__":
    main()