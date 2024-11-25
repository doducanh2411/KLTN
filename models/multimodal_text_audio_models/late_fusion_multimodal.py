import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from transformers import BertModel, AutoTokenizer, ASTForAudioClassification, AutoFeatureExtractor


class MultiModalLateFusionTextAudio(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Video Feature Extractor
        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.cnn.classifier[1] = nn.Linear(
            self.cnn.classifier[1].in_features, 768
        )

        # Text Feature Extractor
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
        # Process video frames
        feature = []
        for t in range(x_3d.size(1)):
            out = self.cnn(x_3d[:, t, :, :, :])
            feature.append(out)

        video_features = torch.mean(torch.stack(feature), dim=0)

        # Process text
        text_token = self.tokenizer(
            title, return_tensors="pt", padding=True, truncation=True
        )
        text_token = {key: value.cuda() for key, value in text_token.items()}
        text_output = self.bert(**text_token)
        text_features = torch.mean(text_output.last_hidden_state, dim=1)

        # Process audio
        audio_features = self.ast(audio).logits

        # Concatenate features
        combined_features = torch.cat(
            (text_features, video_features, audio_features), dim=1
        )

        # Classification
        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.fc2(x)

        return x
