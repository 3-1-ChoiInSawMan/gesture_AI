import numpy as np
import re
import torch
from torch.utils.data import Dataset
import os

class SignDataset(Dataset):
    def __init__(self, npy_dir):
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
        kp = np.load(os.path.join(self.npy_dir, self.files[idx]))  # (120, 137, 3)
        
        # x, y 픽셀 좌표 정규화 (0~1 범위로)
        kp[..., 0] /= 1920.0  # x좌표 / 영상 width
        kp[..., 1] /= 1080.0  # y좌표 / 영상 height
        # confidence(2번째)는 이미 0~1이라 그대로
        
        return torch.FloatTensor(kp), torch.LongTensor([self.labels[idx]])

    def _parse_label(self, filename):
        match = re.search(r'WORD(\d+)', filename)
        return int(match.group(1)) if match else -1