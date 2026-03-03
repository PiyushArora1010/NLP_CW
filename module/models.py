import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(input_dim // 4, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

class ResidualMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        bottleneck_dim=256,
        dropout=0.1,
        use_residual=True,
    ):
        super().__init__()

        self.use_residual = use_residual

        self.norm_in = nn.LayerNorm(input_dim)
        self.norm_out = nn.LayerNorm(input_dim)

        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        residual = self.norm_in(x)

        hidden = self.fc1(residual)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)

        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        if self.use_residual:
            hidden = hidden + residual

        hidden = self.norm_out(hidden)

        logits = self.classifier(hidden)
        return logits

model_registry = {
    "mlp": MLP,
    "residual_mlp": ResidualMLPClassifier,
}