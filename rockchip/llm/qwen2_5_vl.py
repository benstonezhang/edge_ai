import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing_extensions import override

from .base import MultiModalModel


class Qwen2_5_VLVisionForOnnx(torch.nn.Module):
    def __init__(self, vlm, batch_size, height, width):
        super().__init__()
        self.merge_size = 2
        self.temporal_patch_size = 2
        self.patch_size = 14
        self.channel = 3
        self.vpm = vlm.visual
        self.batch_size = batch_size
        self.grid_thw = torch.tensor(
            [[(self.batch_size + self.merge_size - 1) // self.merge_size, height // self.patch_size,
              width // self.patch_size]], dtype=torch.int64)

    def forward(self, pixel_values):
        if self.batch_size == 1:
            patches = pixel_values.repeat(self.temporal_patch_size, 1, 1, 1)
        elif self.batch_size % self.temporal_patch_size == 1:
            repeat_image = pixel_values[-1:, ...].repeat(2, 1, 1, 1)
            patches = torch.cat((pixel_values, repeat_image), dim=0)
        else:
            patches = pixel_values
        grid_t, grid_h, grid_w = self.grid_thw[0][0], self.grid_thw[0][1], self.grid_thw[0][2]
        patches = patches.reshape(grid_t, self.temporal_patch_size, self.channel,
                                  grid_h // self.merge_size, self.merge_size, self.patch_size,
                                  grid_w // self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(grid_t * grid_h * grid_w,
                                          self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)

        return self.vpm(flatten_patches, self.grid_thw)


class Qwen2_5_VLMultiModalModel(MultiModalModel):
    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        self = cls.__new__(cls)
        self.generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            *args, _attn_implementation="eager", **kwargs)
        self.model = self.generation_model.model
        self.batch_size = batch_size
        self.height = height
        self.width = width
        return self

    @override
    def eval(self):
        self.generation_model.eval()
        return self

    @override
    def get_vision(self):
        return Qwen2_5_VLVisionForOnnx(self.model, self.batch_size, self.height, self.width)

    @property
    def vision_mean(self):
        return [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]

    @property
    def vision_std(self):
        return [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]

    @override
    def get_input_embeddings(self, processor: AutoProcessor, text_input: str,
                             image_input=None, audio_input=None, audio_input_mask=None):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", },
                    {"type": "text", "text": text_input},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image_input], padding=True, return_tensors="pt").to(
            self.model.device)
        inputs_embeds = self.model.language_model.embed_tokens(inputs["input_ids"])
        pixel_values = inputs["pixel_values"].type(self.model.visual.dtype)
        image_mask = inputs["input_ids"] == self.model.config.image_token_id
        image_embeds = self.model.visual(pixel_values, grid_thw=inputs["image_grid_thw"]).to(inputs_embeds.device)
        inputs_embeds[image_mask] = image_embeds

        return inputs_embeds
