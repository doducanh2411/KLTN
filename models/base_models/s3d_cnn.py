from torchvision.models.video import s3d, S3D_Weights
import torch.nn as nn
import torch.nn.functional as F


class S3D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = s3d(weights=S3D_Weights.KINETICS400_V1)

        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        # (bs, T, C, H, W) => (bs, C, T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)

        x = self.cnn(x_3d)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
