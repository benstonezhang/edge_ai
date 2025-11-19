import torch
from typing_extensions import override

from .base import MultiModalModel


class InternVLVisionForOnnx(torch.nn.Module):
    def __init__(self, vpm):
        super().__init__()
        self.vpm = vpm

    def forward(self, *args, **kwargs):
        return self.vpm.extract_feature(*args, **kwargs)


class InternVLMultiModalModel(MultiModalModel):
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    vision_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    vision_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image_size = (448, 448)
    image_tokens = '<image>'
    tokenizer_config = {"use_fast": False}
    generation_config = {'max_new_tokens': 1024, 'do_sample': True}

    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        from transformers import AutoModel

        self = cls.__new__(cls)
        self.generation_model = AutoModel.from_pretrained(*args, use_flash_attn=True, **kwargs)
        self.model = self.generation_model.language_model
        self.batch_size = batch_size
        self.height = height
        self.width = width
        return self

    @override
    def get_vision(self):
        return InternVLVisionForOnnx(self.generation_model)

    @override
    def get_input_embeddings(self, processor, text_input: str, image_input: str,
                             audio_input=None, audio_input_mask=None):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", },
                    {"type": "text", "text": text_input},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        text_prompt = self.replace_image_tokens(processor, text_prompt)
        inputs = processor(text=text_prompt, padding=True, return_tensors='pt').to(self.generation_model.device)

        input_ids = inputs['input_ids']
        inputs_embeds = self.generation_model.language_model.get_input_embeddings()(input_ids)

        pixel_values = self.load_image_tensor(image_input).to(self.generation_model.vision_model.device,
                                                              self.generation_model.vision_model.dtype)
        vit_embeds = self.generation_model.extract_feature(pixel_values)

        B, N, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.generation_model.img_context_token_id)
        assert selected.sum() != 0
        inputs_embeds[selected] = vit_embeds.reshape(-1, C).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.reshape(B, N, C)

        return inputs_embeds

    def replace_image_tokens(self, processor: AutoProcessor, text: str):
        self.generation_model.img_context_token_id = processor.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        return text.replace(self.image_tokens,
                            self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.generation_model.num_image_token + self.IMG_END_TOKEN,
                            1)
