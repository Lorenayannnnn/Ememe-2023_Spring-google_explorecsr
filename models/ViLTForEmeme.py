
from torch import nn
from transformers import ViltModel, AutoConfig


class ViLTForMemeSentimentClassification(nn.Module):
    def __init__(self, config: AutoConfig, model_name_or_path: str, model_kwargs: dict):
        super(ViLTForMemeSentimentClassification, self).__init__()
        self.vilt_model = ViltModel.from_pretrained(
            model_name_or_path,
            config=config,
            **model_kwargs
        )

    def forward(self, inputs):
        # return self.vilt_model(**inputs)['pooler_output']
        # just return the ModelOutput with the pooler output
        output = self.vilt_model(**inputs)
        return output, output['pooler_output']
