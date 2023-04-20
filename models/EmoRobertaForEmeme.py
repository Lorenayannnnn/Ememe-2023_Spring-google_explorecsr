
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class EmoRobertaForEmeme(nn.Module):
    def __init__(self, config: AutoConfig, model_name_or_path: str, model_kwargs: dict):
        super(EmoRobertaForEmeme, self).__init__()
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            **model_kwargs
        ).roberta

        self.pooler_layer = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

    def forward(self, inputs):
        outputs = self.roberta_model(**inputs)
        pooled_outputs = self.pooler_layer(outputs.last_hidden_state)

        return outputs, torch.tanh(pooled_outputs[:, -1, :])
