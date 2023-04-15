from typing import Optional, Tuple, Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, ViltProcessor, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import ModelOutput

from models.EmoRobertaForEmeme import EmoRobertaForEmeme
from models.ViLTForEmeme import ViLTForMemeSentimentClassification

class EmemeOutput(ModelOutput):
    """
    from CLIPOutput
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class EmemeModel(PreTrainedModel):
    def __init__(self, config: AutoConfig, model_name_or_path: str, model_kwargs: dict):
        super(self).__init__()
        self.text_model = EmoRobertaForEmeme(
            config=config,
            model_name_or_path=model_name_or_path,
            model_kwargs=model_kwargs
        )
        self.processor = ViltProcessor.from_pretrained(
            model_name_or_path,
            config=config,
            **model_kwargs
        )
        self.meme_model = ViLTForMemeSentimentClassification(
            model_name_or_path=model_name_or_path,
            config=config,
            **model_kwargs
        )
        self.contrastive_loss = self.ContrastiveLoss(self.logit_scale.exp())

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, *model_args, **kwargs)
        model.load_state_dict(torch.load(pretrained_model_name_or_path))
        return model

    class ContrastiveLoss(nn.Module):
        def __init__(self, logit_scale):
            super().__init__()
            self.logit_scale = logit_scale

        def forward(self, text_embeds, image_embeds):
            batch_size = text_embeds.size(0)

            logits_per_text = torch.matmul(text_embeds, image_embeds.T) * self.logit_scale
            logits_per_image = logits_per_text.T
            labels = torch.arange(batch_size).to(logits_per_text.device)

            loss = F.cross_entropy(logits_per_text, labels) + F.cross_entropy(logits_per_text.T, labels)/2
            return loss, logits_per_text, logits_per_image

    def forward(self, inputs):
        text_outputs, text_pooler_output = self.text_model(**inputs)
        image_outputs, image_pooler_output = self.meme_model(**inputs)

        text_embeds = F.normalize(text_pooler_output, dim=1)
        image_embeds = F.normalize(image_pooler_output, dim=1)

        # Compute the contrastive loss
        loss, logits_per_text, logits_per_image = self.contrastive_loss(text_embeds, image_embeds)
        print("loss: ", loss)

        return EmemeOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=image_outputs,
        )
