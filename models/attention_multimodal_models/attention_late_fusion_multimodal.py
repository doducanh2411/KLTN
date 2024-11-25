import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import BertModel, AutoTokenizer, ASTForAudioClassification


class AttentionMultiModalLateFusion(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 768)

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
        feature = []

        for t in range(x_3d.size(1)):
            out = self.cnn(x_3d[:, t, :, :, :])
            feature.append(out)

        video_features = torch.mean(torch.stack(feature), 0)
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
