import torch
import torch.nn as nn
from transformers import VivitForVideoClassification, BertModel, AutoTokenizer


class MultiModalViViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.vivit = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.vivit.classifier = nn.Linear(
            in_features=768, out_features=768, bias=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Linear(768 + 768, num_classes)

    def forward(self, x_3d, title):
        video_output = self.vivit(x_3d)
        video_features = video_output.logits

        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        text_features = torch.mean(text_output.last_hidden_state, dim=1)

        combined_features = torch.cat((text_features, video_features), dim=1)

        x = self.fc(combined_features)

        return x
