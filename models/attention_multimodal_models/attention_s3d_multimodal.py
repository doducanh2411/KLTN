import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import s3d, S3D_Weights
from transformers import BertModel, AutoTokenizer, ASTForAudioClassification


class AttentionMultiModalS3D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = s3d(weights=S3D_Weights.KINETICS400_V1)
        self.cnn.classifier[1] = nn.Conv3d(
            1024, 768, kernel_size=(1, 1, 1), stride=(1, 1, 1))

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
        # (bs, T, C, H, W) => (bs, C, T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)

        video_features = self.cnn(x_3d).unsqueeze(1)  # (batch_size, 1, 768)

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
