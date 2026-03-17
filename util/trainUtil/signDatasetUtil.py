import numpy as np
import re
import torch
from torch.utils.data import Dataset
import os

class SignDataset(Dataset):
    def __init__(self, npy_dir, augment = False):
        self.augment = augment
        self.files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
        self.npy_dir = npy_dir

        raw_labels = [self._parse_label(f) for f in self.files]
        unique = sorted(set(raw_labels))
        self.label2idx = {l: i for i, l in enumerate(unique)}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        self.labels = [self.label2idx[l] for l in raw_labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        kp = np.load(os.path.join(self.npy_dir, self.files[idx]))

        if self.augment:
            # 1. 노이즈 세게
            kp[..., :2] += np.random.normal(0, 0.02, kp[..., :2].shape)

            # 2. 좌우 반전
            if np.random.rand() > 0.5:
                kp[..., 0] = 1.0 - kp[..., 0]

            # 3. 스케일 변환
            scale = np.random.uniform(0.8, 1.2)
            kp[..., :2] *= scale

            # 4. 시간축 드롭 (프레임 일부를 0으로)
            if np.random.rand() > 0.5:
                drop_idx = np.random.choice(120, 20, replace=False)
                kp[drop_idx] = 0

            # 5. 이동
            shift = np.random.uniform(-0.1, 0.1, 2)
            kp[..., :2] += shift

            return torch.FloatTensor(kp), torch.LongTensor([self.labels[idx]])
        
        return torch.FloatTensor(kp), torch.LongTensor([self.labels[idx]])

    def _parse_label(self, filename):
        match = re.search(r'WORD(\d+)', filename)
        return int(match.group(1)) if match else -1

class SignDatasetSmall(SignDataset):
    def __init__(self, npy_dir, max_classes=10):
        super().__init__(npy_dir)
        
        # 앞 10개 클래스만 필터
        valid_labels = set(sorted(self.label2idx.keys())[:max_classes])
        filtered = [(f, l) for f, l in zip(self.files, self.labels) 
                    if list(self.label2idx.keys())[list(self.label2idx.values()).index(l)] in valid_labels]
        
        self.files  = [f for f, l in filtered]
        self.labels = [l for f, l in filtered]
        
        print(f"필터된 샘플: {len(self.files)}개")
        print(f"클래스: {sorted(set(self.labels))}")