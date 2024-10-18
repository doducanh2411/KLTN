from torchvision.models.video import s3d, S3D_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch


class S3D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = s3d(weights=S3D_Weights.KINETICS400_V1)
        self.cnn.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Conv3d(1024, 512, kernel_size=(
                1, 1, 1), stride=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(128, num_classes, kernel_size=(
                1, 1, 1), stride=(1, 1, 1))
        )

    def forward(self, x_3d):
        # (bs, T, H, W, C) => (bs, C, T, H, W)
        x_3d = x_3d.permute(0, 4, 1, 2, 3)

        x = self.cnn(x_3d)

        return x
