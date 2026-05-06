import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 설정
# =========================
DATASET_DIR = "dataset"   # dataset/class_name/*.npy
SEQ_LEN = 30
INPUT_DIM = 88
BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 시드 고정
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 데이터 분할
# 클래스당 30개 기준: 24 / 3 / 3
# =========================
def build_split(dataset_dir):
    class_names = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )

    label2idx = {name: i for i, name in enumerate(class_names)}
    idx2label = {i: name for name, i in label2idx.items()}

    train_samples = []
    val_samples = []
    test_samples = []

    for cls in class_names:
        cls_dir = os.path.join(dataset_dir, cls)
        files = sorted(
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.endswith(".npy")
        )

        random.shuffle(files)

        n = len(files)
        if n < 10:
            raise ValueError(f"{cls} 클래스 샘플이 너무 적음: {n}")

        # 대충 80 / 10 / 10
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val

        # 너무 작게 쪼개지는 거 방지
        if n_val < 1:
            n_val = 1
            n_train -= 1
        if n_test < 1:
            n_test = 1
            n_train -= 1

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        train_samples += [(p, label2idx[cls]) for p in train_files]
        val_samples += [(p, label2idx[cls]) for p in val_files]
        test_samples += [(p, label2idx[cls]) for p in test_files]

    return train_samples, val_samples, test_samples, label2idx, idx2label


# =========================
# Dataset
# =========================
class GestureDataset(Dataset):
    def __init__(self, samples, seq_len=30, augment=False):
        self.samples = samples
        self.seq_len = seq_len
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _fix_length(self, x):
        # x: (T, 88)
        T = x.shape[0]

        if T == self.seq_len:
            return x

        if T > self.seq_len:
            idx = np.linspace(0, T - 1, self.seq_len).astype(int)
            return x[idx]

        pad_len = self.seq_len - T
        pad = np.repeat(x[-1][None, :], pad_len, axis=0)
        return np.concatenate([x, pad], axis=0)

    def _normalize(self, x):
        # sample-wise normalization
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-6
        return (x - mean) / std

    def _augment(self, x):
        # 너무 세게 하지 말 것
        noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
        scale = np.random.uniform(0.98, 1.02)
        x = x * scale + noise
        return x

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)

        if x.ndim != 2:
            raise ValueError(f"입력 shape 이상함: {path}, shape={x.shape}")

        if x.shape[1] != INPUT_DIM:
            raise ValueError(f"feature dim이 88이 아님: {path}, shape={x.shape}")

        x = self._fix_length(x)
        x = self._normalize(x)

        if self.augment:
            x = self._augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# =========================
# 모델
# =========================
class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 60, 88)
        out, _ = self.gru(x)
        last = out[:, -1, :]   # (B, hidden_dim*2)
        return self.head(last)


# =========================
# 평가
# =========================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    return total_loss / total, correct / total, y_true, y_pred


# =========================
# 학습
# =========================
def main():
    set_seed(SEED)

    train_samples, val_samples, test_samples, label2idx, idx2label = build_split(DATASET_DIR)

    print("classes:", label2idx)
    print("train:", len(train_samples), "val:", len(val_samples), "test:", len(test_samples))

    train_ds = GestureDataset(train_samples, seq_len=SEQ_LEN, augment=True)
    val_ds = GestureDataset(val_samples, seq_len=SEQ_LEN, augment=False)
    test_ds = GestureDataset(test_samples, seq_len=SEQ_LEN, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BiGRUClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(label2idx),
        dropout=0.3,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    patience = 12
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        print(
            f"[{epoch:03d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "seq_len": SEQ_LEN,
                    "input_dim": INPUT_DIM,
                },
                "best_bigru.pt",
            )
            print("best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("early stopping")
            break

    # 테스트
    ckpt = torch.load("best_bigru.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)

    print("\n[TEST]")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc ={test_acc:.4f}")

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(idx2label))], zero_division=0))

        print("\n[Confusion Matrix]")
        print(confusion_matrix(y_true, y_pred))
    except Exception as e:
        print("sklearn report skip:", e)


if __name__ == "__main__":
    main()