import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=256,
                            num_layers=3, bidirectional=True)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            out = self.cnn(x_3d[:, t, :, :, :])
            out = out.unsqueeze(0)
            out, hidden = self.lstm(out, hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)

        return x
