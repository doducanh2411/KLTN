import torch
import torch.nn as nn
from transformers import VivitForVideoClassification, BertModel, AutoTokenizer


class MultiModalViViT(nn.Module):
    def __init__(self, num_classes=4, embed_dim=768, num_heads=8):
        super().__init__()

        # Pre-trained models
        self.vivit = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.vivit.classifier = nn.Linear(
            in_features=768, out_features=768, bias=True)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,  # Enables (batch, seq, feature) format
        )

        # Classification layer
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x_3d, title):
        # Video features
        video_output = self.vivit(x_3d)
        video_features = video_output.logits.unsqueeze(
            1)  # (batch_size, 1, embed_dim)

        # Text features
        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True
        )
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        # (batch_size, seq_len, embed_dim)
        text_features = text_output.last_hidden_state

        # Concatenate video and text as a single sequence
        # (batch_size, seq_len + 1, embed_dim)
        combined_features = torch.cat((video_features, text_features), dim=1)

        # Multi-Head Attention
        attn_output, _ = self.multihead_attn(
            query=combined_features,
            key=combined_features,
            value=combined_features,
        )  # (batch_size, seq_len + 1, embed_dim)

        # Use the first token (video feature after attention) for classification
        cls_token = attn_output[:, 0, :]  # (batch_size, embed_dim)

        # Classification
        x = self.fc(cls_token)  # (batch_size, num_classes)

        return x
