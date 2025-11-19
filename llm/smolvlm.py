import torch
from typing_extensions import override

from .base import MultiModalModel


class SmolVLMVisionForOnnx(torch.nn.Module):
    def __init__(self, vlm):
        super(SmolVLMVisionForOnnx, self).__init__()
        self.vlm = vlm

    def forward(self, *args, **kwargs):
        return self.vlm.extract_feature(*args, **kwargs)


class SmolVLMMultiModalModel(MultiModalModel):
    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        from transformers import SmolVLMForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = SmolVLMForConditionalGeneration.from_pretrained(
                *args, _attn_implementation='eager', **kwargs)
        self.model = self.generation_model.model
        return self

    @override
    def get_vision(self):
        return SmolVLMVisionForOnnx(self.model)

    @override
    def get_input_embeddings(self, processor, text_input: str, image_input: str,
                             audio_input=None, audio_input_mask=None):
        from PIL import Image

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_input},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.open(image_input)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(
                self.model.device)
        inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        pixel_values = inputs["pixel_values"].type(self.model.visual.get_dtype())
        image_mask = inputs["input_ids"] == self.model.config.image_token_id
        image_embeds = self.model.visual(pixel_values, grid_thw=inputs["image_grid_thw"]).to(inputs_embeds.device)
        inputs_embeds[image_mask] = image_embeds

        return inputs_embeds
