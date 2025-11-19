import torch
from typing_extensions import override

from .base import MultiModalModel
from .qwen2_vl import Qwen2_VLVisionForOnnx


class Qwen2_5_VLVisionForOnnx(Qwen2_VLVisionForOnnx):
    def __init__(self, vpm, batch_size, height, width):
        super().__init__(vpm, batch_size, height, width)

    @override
    def forward(self, pixel_values):
        if self.batch_size == 1:
            patches = pixel_values.repeat(self.temporal_patch_size, 1, 1, 1)
        elif self.batch_size % self.temporal_patch_size == 1:
            repeat_image = pixel_values[-1:, ...].repeat(2, 1, 1, 1)
            patches = torch.cat((pixel_values, repeat_image), dim=0)
        else:
            patches = pixel_values
        patches = patches.reshape(self.grid_t, self.temporal_patch_size, self.channel,
                                  self.grid_h // self.merge_size, self.merge_size, self.patch_size,
                                  self.grid_w // self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(self.grid_t * self.grid_h * self.grid_w,
                                          self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
        return self.vpm(flatten_patches, self.grid_thw)


class Qwen2_5_VLMultiModalModel(MultiModalModel):
    vision_mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    vision_std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
    image_size = (448, 448)
    tokenizer_config = {"min_pixels": 256 * 28 * 28, "max_pixels": 2048 * 28 * 28}

    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                *args, _attn_implementation="eager", **kwargs)
        self.model = self.generation_model.model
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.vision_model = self.generation_model.visual if hasattr(self.generation_model, 'visual') \
            else self.model.visual
        self.embed_tokens = self.model.embed_tokens if hasattr(self.model, 'embed_tokens') \
            else self.model.language_model.embed_tokens
        return self

    @override
    def get_vision(self):
        return Qwen2_5_VLVisionForOnnx(self.vision_model, self.batch_size, self.height, self.width)

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

        # Preprocess the inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        image = Image.open(image_input)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(self.model.device)

        inputs_embeds = self.embed_tokens(inputs["input_ids"])
        pixel_values = inputs["pixel_values"].type(self.vision_model.dtype)
        image_mask = inputs["input_ids"] == self.model.config.image_token_id
        image_embeds = self.vision_model(pixel_values, grid_thw=inputs["image_grid_thw"]).to(inputs_embeds.device)
        inputs_embeds[image_mask] = image_embeds

        return inputs_embeds
