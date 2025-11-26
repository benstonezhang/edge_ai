import torch
from typing_extensions import override

from .model import MultiModalModel


class SmolVLMVisionForOnnx(torch.nn.Module):
    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm.vision_model
        self.connector = vlm.connector

    def forward(self, pixel_values):
        # Get sequence from the vision encoder
        image_hidden_states = self.vlm(pixel_values).last_hidden_state
        # Modality projection & resampling
        image_hidden_states = self.connector(image_hidden_states)
        print("image_features:", image_hidden_states.shape)
        return image_hidden_states


class SmolVLMMultiModalModel(MultiModalModel):
    image_size = 672
    # image_tokens = '<image>'
    processor_config = {"use_fast": False}
    output_need_trim = True

    @override
    @classmethod
    def from_pretrained(cls, model_name: str, load_processor: bool, *args, **kwargs):
        from transformers import SmolVLMForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = SmolVLMForConditionalGeneration.from_pretrained(
                model_name, *args, _attn_implementation='eager', **kwargs)
        self.model = self.generation_model.model
        if load_processor:
            from transformers import SmolVLMProcessor

            self.processor = SmolVLMProcessor.from_pretrained(model_name, **self.processor_config)
        return self

    @override
    def get_vision(self):
        return SmolVLMVisionForOnnx(self.model)

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
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.open(image_input)
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(
                self.model.device)
        input_ids = inputs["input_ids"]

        inputs_embeds = self.model.text_model.get_input_embeddings()(input_ids).to(self.model.device)

        pixel_values = inputs["pixel_values"].type(self.model.vision_model.dtype)
        image_embeds = self.model.get_image_features(pixel_values).to(inputs_embeds.device)

        inputs_embeds = self.model.inputs_merger(input_ids=input_ids,
                                                 inputs_embeds=inputs_embeds,
                                                 image_hidden_states=image_embeds)

        return inputs_embeds

    @override
    def generate(self, **inputs):
        generate_ids = self.generation_model.generate(**inputs)
        input_len = inputs["input_ids"].shape[-1]
        return generate_ids[0][input_len:].unsqueeze(0)
