from transformers import VideoMAEForVideoClassification
from torch import nn
import torch


class VideoMAE(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.mae = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics")
        self.mae.classifier = nn.Linear(
            in_features=768, out_features=num_classes, bias=True)

    def forward(self, x_3d):
        x = self.mae(x_3d)

        return x.logits