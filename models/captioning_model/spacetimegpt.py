import torch.nn as nn
import torch
from transformers import AutoTokenizer, VisionEncoderDecoderModel


class SpaceTimeGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "Neleac/timesformer-gpt2-video-captioning").to(self.device)

        self.gen_kwargs = {
            "min_length": 10,
            "max_length": 20,
            "num_beams": 8,
        }

    def forward(self, x_3d):
        tokens = self.model.generate(x_3d, **self.gen_kwargs)

        caption = self.tokenizer.batch_decode(
            tokens, skip_special_tokens=True)[0]

        return caption
