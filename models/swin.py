import torch.nn as nn
from torchvision.models.video import swin3d_b, Swin3D_B_Weights


class Swin(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.swin = swin3d_b(
            weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        self.swin.head = nn.Linear(
            in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x_3d):
        # (bs, T, H, W, C) => (bs, T, C, H, W)
        x_3d = x_3d.permute(0, 4, 1, 2, 3)

        x = self.swin(x_3d)

        return x
