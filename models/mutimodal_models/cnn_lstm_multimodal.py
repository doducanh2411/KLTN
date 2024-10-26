import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import BertModel, AutoTokenizer


class MultiModalCNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=256,
                            num_layers=3, bidirectional=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Linear(512 + 768, num_classes)

    def forward(self, x_3d, title):
        hidden = None
        for t in range(x_3d.size(1)):
            out = self.cnn(x_3d[:, t, :, :, :])
            out = out.unsqueeze(0)
            out, hidden = self.lstm(out, hidden)

        video_features = out[-1, :, :]

        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True)
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        text_features = torch.mean(text_output.last_hidden_state, dim=1)

        combined_features = torch.cat((text_features, video_features), dim=1)

        x = self.fc(combined_features)

        return x
