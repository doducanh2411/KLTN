from transformers import VivitForVideoClassification
from torch import nn


class ViViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.vivit = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.vivit.classifier = nn.Linear(
            in_features=768, out_features=num_classes, bias=True)

    def forward(self, x_3d):
        x = self.vivit(x_3d)

        return x.logits
