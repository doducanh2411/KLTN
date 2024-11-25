import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitForVideoClassification, BertModel, AutoTokenizer, ASTForAudioClassification


class MultiModalTextAudioViViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Video Feature Extractor
        self.vivit = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.vivit.classifier = nn.Linear(
            in_features=768, out_features=768, bias=True)

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
        video_output = self.vivit(x_3d)
        video_features = video_output.logits

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
