import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_b, Swin3D_B_Weights


class Swin(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.swin = swin3d_b(
            weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        self.swin.head = nn.Linear(
            in_features=1024, out_features=512, bias=True)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        # (bs, T, H, W, C) => (bs, T, C, H, W)
        x_3d = x_3d.permute(0, 4, 1, 2, 3)

        x = self.swin(x_3d)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
