from transformers import VivitForVideoClassification, VivitConfig
from torch import nn
import torch.nn.functional as F


class ViViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.config = VivitConfig(image_size=112, num_frames=32)
        self.vivit = VivitForVideoClassification(config=self.config)
        self.vivit.classifier = nn.Linear(
            in_features=768, out_features=num_classes, bias=True)

    def forward(self, x_3d):
        x = self.vivit(x_3d)

        return x.logits
