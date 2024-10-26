import torch
import torch.nn as nn
from torchvision.models.video import s3d, S3D_Weights
from transformers import BertModel, AutoTokenizer


class MultiModalS3D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = s3d(weights=S3D_Weights.KINETICS400_V1)
        self.cnn.avgpool = nn.AvgPool3d(
            kernel_size=(2, 3, 3), stride=1, padding=0)
        self.cnn.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Conv3d(1024, 768, kernel_size=(
                1, 1, 1), stride=(1, 1, 1)),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Linear(768 + 768, num_classes)

    def forward(self, x_3d, title):
        # (BS, T, C, H, W) -> (BS, C, T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)

        video_features = self.cnn(x_3d)

        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        text_features = torch.mean(text_output.last_hidden_state, dim=1)

        combined_features = torch.cat((text_features, video_features), dim=1)

        x = self.fc(combined_features)

        return x
