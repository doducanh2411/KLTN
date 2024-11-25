import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import s3d, S3D_Weights
from transformers import BertModel, AutoTokenizer, ASTForAudioClassification


class MultiModalTextAudioS3D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        # Video Feature Extractor
        self.cnn = s3d(weights=S3D_Weights.KINETICS400_V1)
        self.cnn.classifier[1] = nn.Conv3d(
            1024, 768, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        # Text Feature Extractor
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Audio Feature Extractor
        self.ast = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.ast.classifier.dense = nn.Linear(
            in_features=768, out_features=768, bias=True
        )

        self.fc1 = nn.Linear(768 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d, title, audio):
        # Process video
        # (bs, T, C, H, W) => (bs, C, T, H, W)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)

        video_features = self.cnn(x_3d)

        # Process text
        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        text_features = torch.mean(text_output.last_hidden_state, dim=1)

        # Process audio
        audio_features = self.ast(audio).logits

        # Concatenate features
        combined_features = torch.cat(
            (text_features, video_features, audio_features), dim=1)

        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.fc2(x)

        return x
