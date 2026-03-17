import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from util.trainUtil.signDatasetUtil import SignDataset, SignDatasetSmall
from util.trainUtil.signTransformerUtil import SignTransformer


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 데이터셋
    for n_classes in [100]:
        dataset = SignDatasetSmall("/data/word_npy_norm", max_classes=n_classes)
    from sklearn.model_selection import train_test_split
    # 클래스 균형 맞춰서 분리
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=dataset.labels,  # 클래스 비율 유지
        random_state=42
    )

    train_set = torch.utils.data.Subset(
        SignDataset("/data/word_npy_norm", augment=True), train_idx
    )
    val_set = torch.utils.data.Subset(
        SignDataset("/data/word_npy_norm", augment=False), val_idx
    )

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # 모델
    model     = SignTransformer(num_classes=100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        epochs=200,
        steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(200):
        # train
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.squeeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.squeeze(1).to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total   += y.size(0)

        val_acc  = correct / total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1:03d} | loss: {avg_loss:.4f} | val_acc: {val_acc:.4f}")

        # 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'label2idx':        dataset.label2idx,
                'idx2label':        dataset.idx2label,
                'val_acc':          val_acc,
            }, "best_model.pt")
            print(f"  >>> best model 저장 (val_acc: {val_acc:.4f})")

    print(f"\n학습 완료. best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
