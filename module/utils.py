import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

def getData(file_path, split_file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    data = [line.strip().split('\t') for line in data[4:]]
    data = [
        {
            "id": int(line[0]),
            "text": str(line[4]),
            "label": 1 if any(int(line[5]) == i for i in [2,3,4]) else 0
        }
        for line in data
    ]

    split_file = pd.read_csv(split_file_path).set_index("par_id")
    split_ids = set(split_file.index)
    
    data = [d for d in data if d["id"] in split_ids]
    return data

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor like [w0, w1] or None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # pt = probability of the true class
        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss

        return focal_loss.mean()