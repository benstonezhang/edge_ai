import torch
from transformers import Gemma3ForConditionalGeneration, Gemma3Model, AutoProcessor
from typing_extensions import override

from .base import MultiModalModel


class Gemma3VisionForOnnx(torch.nn.Module):
    def __init__(self, vlm: Gemma3Model):
        super().__init__()
        self.vpm = vlm

    def forward(self, pixel_values):
        return self.vlm.get_image_features(pixel_values)


class Gemma3MultiModalModel(MultiModalModel):
    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        self = cls.__new__(cls)
        self.generation_model = Gemma3ForConditionalGeneration.from_pretrained(*args, **kwargs)
        self.model = self.generation_model.model
        return self

    @override
    def eval(self):
        self.generation_model.eval()
        return self

    @override
    def get_vision(self):
        return Gemma3VisionForOnnx(self.model)

    @property
    def vision_mean(self):
        return [123.675, 116.28, 103.53]

    @property
    def vision_std(self):
        return [58.395, 58.395, 58.395]

    @override
    def get_input_embeddings(self, processor: AutoProcessor, text_input: str,
                             image_input=None, audio_input=None, audio_input_mask=None):
        inputs = processor(text='<start_of_image> ' + text_input,
                           images=[image_input],
                           padding=True,
                           return_tensors='pt').to(self.model.device)

        input_ids = inputs['input_ids']

        # Replace image id with PAD if the image token is OOV, to avoid index-errors
        if input_ids is not None and self.model.config.image_token_id >= self.model.vocab_size:
            special_image_mask = input_ids == self.model.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids
        inputs_embeds = self.model.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if image_input is not None:
            pixel_values = inputs['pixel_values']
            image_features = self.model.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                                 image_features=image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds
