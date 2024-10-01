from transformers import VivitForVideoClassification, VivitConfig
from torch import nn


class ViViT(nn.Module):
    def __init__(self, num_classes=4, image_size=112, num_frames=900):
        super.__init__()
        self.config = VivitConfig()
        self.config.image_size = image_size
        self.config.num_frames = num_frames
        self.config.num_labels = num_classes

        self.vivit = VivitForVideoClassification(self.config)

    def forward(self, x_3d):
        # (bs, T, H, W, C) => (bs, T, C, H, W)
        x_3d = x_3d.permute(0, 1, 4, 2, 3)

        x = self.vivit(x_3d)

        return x
