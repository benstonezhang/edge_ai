from typing import override

import torch

from .model import MultiModalModel


class Qwen_VLVisionForOnnx(torch.nn.Module):
    merge_size = 2
    temporal_patch_size = 2
    patch_size = 14
    channel = 3

    def __init__(self, vpm, batch_size, height, width, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vpm = vpm
        self.batch_size = batch_size
        self.grid_t = (batch_size + self.temporal_patch_size - 1) // self.temporal_patch_size
        self.grid_h = height // self.patch_size
        self.grid_w = width // self.patch_size

    def forward(self, pixel_values):
        pixel_values = pixel_values.type(self.vpm.dtype)
        if self.batch_size == 1:
            pixel_inputs = pixel_values.repeat(self.temporal_patch_size, 1, 1, 1)
        elif self.batch_size % self.temporal_patch_size != 0:
            repeat_time = self.temporal_patch_size - self.batch_size % self.temporal_patch_size
            repeat_image = pixel_values[-1:, ...].repeat(repeat_time, 1, 1, 1)
            pixel_inputs = torch.cat((pixel_values, repeat_image), dim=0)
        else:
            pixel_inputs = pixel_values
        patches = pixel_inputs.reshape(self.grid_t, self.temporal_patch_size, self.channel,
                                       self.grid_h // self.merge_size, self.merge_size, self.patch_size,
                                       self.grid_w // self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(self.grid_t * self.grid_h * self.grid_w,
                                          self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
        return self.vpm(flatten_patches)


# The default range for the number of visual tokens per image in the model is 4-16384.You can set min_pixels and
# max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

class Qwen_VLMultiModalModel(MultiModalModel):
    vision_mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    vision_std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
    image_size = 448
    # image_tokens = '<|vision_start|><|image_pad|>...<|vision_end|>'
    processor_config = dict(min_pixels=256 * 28 * 28, max_pixels=2048 * 28 * 28)

    def __init__(self):
        super().__init__()
        self.model = None
        self.vision_model = None
        self.embed_tokens = None

    @override
    def get_input_embeddings(self, text_input: str, image_input: str, audio_input=None, audio_input_mask=None):
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
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        image = Image.open(image_input)
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(
                self.model.device)

        inputs_embeds = self.embed_tokens(inputs["input_ids"])
        pixel_values = inputs["pixel_values"].type(self.vision_model.dtype)
        image_mask = inputs["input_ids"] == self.model.config.image_token_id
        image_embeds, _ = self.vision_model(pixel_values, grid_thw=inputs["image_grid_thw"])
        inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.device)

        return inputs_embeds
