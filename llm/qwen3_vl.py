import torch
from typing_extensions import override

from .qwen_vl import Qwen_VLMultiModalModel, Qwen_VLVisionForOnnx


class Qwen3_VLVisionForOnnx(Qwen_VLVisionForOnnx):
    patch_size = 16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vision_forward = self.vpm.forward

        def forward(pixel_values):
            grid_thw = torch.asarray([[self.grid_t, self.grid_h, self.grid_w]])
            return vision_forward(pixel_values, grid_thw)

        self.vpm.forward = forward


class Qwen3_VLMultiModalModel(Qwen_VLMultiModalModel):
    dynamo_compatible = False

    @override
    @classmethod
    def from_pretrained(cls, model_name: str, load_processor: bool, *args, **kwargs):
        from transformers import Qwen3VLForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, *args, _attn_implementation="eager", **kwargs)
        self.model = self.generation_model.model
        self.vision_model = self.model.visual
        self.embed_tokens = self.model.language_model.embed_tokens
        if load_processor:
            from transformers import Qwen3VLProcessor

            self.processor = Qwen3VLProcessor.from_pretrained(model_name, **self.processor_config)
        return self

    @override
    def get_vision(self):
        return Qwen3_VLVisionForOnnx(self.vision_model, 1, self.image_size, self.image_size)
