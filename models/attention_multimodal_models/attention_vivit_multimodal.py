import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitForVideoClassification, BertModel, AutoTokenizer, ASTForAudioClassification


class AttentionMultiModalViViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.vivit = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.vivit.classifier = nn.Linear(
            in_features=768, out_features=768, bias=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.ast.classifier.dense = nn.Linear(
            in_features=768, out_features=768, bias=True
        )

        self.video_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True)
        self.text_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True)
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, batch_first=True)

        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d, title, audio):
        video_output = self.vivit(x_3d)
        video_features = video_output.logits
        video_features = video_features.unsqueeze(1)  # (batch_size, 1, 768)

        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        # (batch_size, seq_len, 768)
        text_features = text_output.last_hidden_state

        audio_features = self.ast(audio).logits
        audio_features = audio_features.unsqueeze(1)  # (batch_size, 1, 768)

        attention_video, _ = self.video_attention(
            video_features, video_features, video_features)

        attention_text, _ = self.text_attention(
            text_features, text_features, text_features)

        attention_audio, _ = self.audio_attention(
            audio_features, audio_features, audio_features)

        combined_features = torch.cat(
            (attention_video, attention_text, attention_audio), dim=1)
        pooled_features = combined_features.mean(dim=1)  # (batch_size, 768)

        x = self.fc1(pooled_features)
        x = F.relu(x)
        x = self.fc2(x)
        return x
