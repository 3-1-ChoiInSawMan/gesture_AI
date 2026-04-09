import torch
import torch.nn as nn

class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=128, num_layers=2, num_classes=29, dropout=0.3):
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
