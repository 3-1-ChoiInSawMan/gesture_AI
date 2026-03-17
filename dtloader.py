import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset

class SignDataset(Dataset):
    def __init__(self, npy_dir):
        self.files = sorted([
            f for f in os.listdir(npy_dir) if f.endswith('.npy')
        ])
        self.npy_dir = npy_dir

        # 라벨 추출 & 정수 인덱스로 변환
        raw_labels = [self._parse_label(f) for f in self.files]
        unique = sorted(set(raw_labels))
        self.label2idx = {l: i for i, l in enumerate(unique)}
        self.labels = [self.label2idx[l] for l in raw_labels]

        print(f"총 샘플: {len(self.files)}개")
        print(f"클래스 수: {len(unique)}개")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        kp = np.load(os.path.join(self.npy_dir, self.files[idx]))
        return torch.FloatTensor(kp), torch.LongTensor([self.labels[idx]])

    def _parse_label(self, filename):
        match = re.search(r'WORD(\d+)', filename)
        return int(match.group(1)) if match else -1


dataset = SignDataset("/home/ingyu/004.수어영상/preprocessed/word_npy")