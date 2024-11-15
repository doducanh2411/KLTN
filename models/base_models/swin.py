import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import swin3d_s, Swin3D_S_Weights


class Swin(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.swin = swin3d_s(
            weights=Swin3D_S_Weights.KINETICS400_V1)
        self.swin.head = nn.Linear(
            in_features=768, out_features=512, bias=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        # (BS, T, C, H, W) -> (BS, C, T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)
        out = self.swin(x_3d)

        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)

        return x
