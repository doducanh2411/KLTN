import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class SingleFrame(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 512)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        logits = []

        for t in range(x_3d.size(1)):
            x = self.cnn(x_3d[:, t, :, :, :])

            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)

            logits.append(x)

        logits = torch.stack(logits, dim=1)
        logits = torch.mean(logits, dim=1)

        return logits
