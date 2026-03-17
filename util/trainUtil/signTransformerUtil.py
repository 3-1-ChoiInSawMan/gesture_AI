import torch.nn as nn

class SignTransformer(nn.Module):
    def __init__(self, input_dim=411, num_classes=3000,
                 d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):            # (batch, 120, 137, 3)
        x = x.flatten(start_dim=2)   # (batch, 120, 411)
        x = self.embedding(x)         # (batch, 120, 256)
        x = self.transformer(x)       # (batch, 120, 256)
        x = x.mean(dim=1)             # (batch, 256)
        return self.classifier(x)     # (batch, 3000)
    


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=411*120, num_classes=3000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):              # (batch, 120, 137, 3)
        x = x.flatten(start_dim=1)     # (batch, 120*411)
        return self.net(x)