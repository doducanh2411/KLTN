import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class EarlyFusion(nn.Module):
    def __init__(self, num_classes=4, num_input_channels=32):
        super().__init__()

        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.features[0][0] = nn.Conv2d(num_input_channels * 3, 32, kernel_size=(
            3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 512)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        # (bs, T*C, H, W)
        x_3d = x_3d.view(x_3d.size(0), x_3d.size(
            1) * x_3d.size(2), x_3d.size(3), x_3d.size(4))

        out = self.cnn(x_3d)

        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)

        return x
