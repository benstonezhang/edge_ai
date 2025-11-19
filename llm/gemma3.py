import torch
from typing_extensions import override

from .base import MultiModalModel


class Gemma3VisionForOnnx(torch.nn.Module):

    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm

    def forward(self, pixel_values):
        return self.vlm.get_image_features(pixel_values)


class Gemma3MultiModalModel(MultiModalModel):
    vision_mean = [123.675, 116.28, 103.53]
    vision_std = [58.395, 58.395, 58.395]
    image_size = (896, 896)
    generation_config = {'max_new_tokens': 1024, 'do_sample': False}
    output_prefix_with_prompt = True

    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        from transformers import Gemma3ForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = Gemma3ForConditionalGeneration.from_pretrained(*args, **kwargs)
        self.model = self.generation_model.model if hasattr(self.generation_model, 'model') else self.generation_model

        return self

    @override
    def get_vision(self):
        return Gemma3VisionForOnnx(self.model)

    @override
    def get_input_embeddings(self, processor, text_input: str, image_input: str,
                             audio_input=None, audio_input_mask=None):
        from PIL import Image

        image = Image.open(image_input)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_input}
                ]
            }
        ]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        input_ids = inputs['input_ids']

        # Replace image id with PAD if the image token is OOV, to avoid index-errors
        if processor.image_token_id >= self.model.vocab_size:
            special_image_mask = input_ids == processor.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids
        inputs_embeds = self.model.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        pixel_values = inputs['pixel_values']
        image_features = self.model.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
        if hasattr(self.model, 'get_placeholder_mask'):
            special_image_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                                 image_features=image_features)
        else:
            special_image_mask = (input_ids == processor.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds
