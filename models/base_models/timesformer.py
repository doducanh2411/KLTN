from transformers import TimesformerForVideoClassification
from torch import nn


class Timesformer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.timesformer = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400")
        self.timesformer.classifier = nn.Linear(
            in_features=768, out_features=num_classes, bias=True)

    def forward(self, x_3d):
        x = self.timesformer(x_3d)

        return x.logits
