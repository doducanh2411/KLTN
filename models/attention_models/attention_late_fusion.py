import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import BertModel, AutoTokenizer


class AttentionLateFusion(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 768)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True)

        self.fc = nn.Linear(768, num_classes)

    def forward(self, x_3d, title):
        feature = []

        for t in range(x_3d.size(1)):
            out = self.cnn(x_3d[:, t, :, :, :])
            feature.append(out)

        # (batch_size, num_frames, 768)
        video_features = torch.stack(feature, dim=1)

        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        # (batch_size, seq_len, 768)
        text_features = text_output.last_hidden_state

        attn_output, _ = self.cross_attention(
            query=text_features, key=video_features, value=video_features)

        x = attn_output.mean(dim=1)  # (batch_size, 768)
        x = self.fc(x)

        return x
