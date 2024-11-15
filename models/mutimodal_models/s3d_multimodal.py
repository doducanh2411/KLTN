import torch
import torch.nn as nn
from torchvision.models.video import s3d, S3D_Weights
from transformers import BertModel, AutoTokenizer


class MultiModalS3D(nn.Module):
    def __init__(self, num_classes=4, embed_dim=768, num_heads=8):
        super().__init__()

        # Pretrained S3D model
        self.cnn = s3d(weights=S3D_Weights.KINETICS400_V1)
        self.cnn.avgpool = nn.AvgPool3d(
            kernel_size=(2, 3, 3), stride=1, padding=0)
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv3d(1024, embed_dim, kernel_size=(
                1, 1, 1), stride=(1, 1, 1)),
        )

        # Pretrained BERT model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Multi-Head Attention for combining video and text features
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x_3d, title):
        # Video features
        # (BS, T, C, H, W) -> (BS, C, T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)
        video_features = self.cnn(x_3d)  # (BS, embed_dim, 1, 1, 1)
        # (BS, embed_dim)
        video_features = video_features.squeeze(-1).squeeze(-1).squeeze(-1)
        video_features = video_features.unsqueeze(1)  # (BS, 1, embed_dim)

        # Text features
        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        # (BS, seq_len, embed_dim)
        text_features = text_output.last_hidden_state

        # Concatenate video features with text features
        # (BS, seq_len + 1, embed_dim)
        combined_features = torch.cat((video_features, text_features), dim=1)

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(
            query=combined_features,
            key=combined_features,
            value=combined_features,
        )  # (BS, seq_len + 1, embed_dim)

        # Use the first token (video feature after attention) for classification
        cls_token = attn_output[:, 0, :]  # (BS, embed_dim)

        # Classification
        x = self.fc(cls_token)  # (BS, num_classes)

        return x
