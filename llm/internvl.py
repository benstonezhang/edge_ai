from torch.nn import Module
from typing_extensions import override

from .model import MultiModalModel


class InternVLVisionForOnnx(Module):
    def __init__(self, vpm):
        super().__init__()
        self.vpm = vpm
        self.forward = self.vpm.extract_feature


class InternVLMultiModalModel(MultiModalModel):
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    vision_mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    vision_std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image_size = 448
    image_tokens = '<image>'
    processor_config = {"use_fast": False}
    generation_config = {'max_new_tokens': 1024, 'do_sample': True}

    @override
    @classmethod
    def from_pretrained(cls, model_name: str, load_processor: bool, *args, **kwargs):
        from transformers import AutoModel

        self = cls.__new__(cls)
        self.generation_model = AutoModel.from_pretrained(model_name, *args, use_flash_attn=True,
                                                          trust_remote_code=True, **kwargs)
        self.model = self.generation_model.language_model
        if load_processor:
            from torchvision.transforms.functional import pil_to_tensor
            from transformers import AutoProcessor
            from transformers.image_utils import make_flat_list_of_images

            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **self.processor_config)
            _encode_plus = self.processor.encode_plus

            def encode_plus(text: str, images, *_args, **_kwargs):
                text = text.replace(self.image_tokens,
                                    self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.generation_model.num_image_token + self.IMG_END_TOKEN,
                                    1)
                inputs = _encode_plus(text, *_args, **_kwargs)
                images = make_flat_list_of_images(images)
                inputs['pixel_values'] = pil_to_tensor(images[0]).unsqueeze(0).to(self.generation_model.device,
                                                                                  self.generation_model.dtype)
                return inputs

            self.processor.encode_plus = encode_plus
        return self

    @override
    def get_vision(self):
        if hasattr(self.generation_model, 'get_image_features'):
            from .model import VisionModelForOnnx

            return VisionModelForOnnx(self.generation_model)
        else:
            return InternVLVisionForOnnx(self.generation_model)

    @override
    def get_input_embeddings(self, text_input: str, image_input: str, audio_input=None, audio_input_mask=None):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", },
                    {"type": "text", "text": text_input},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        text_prompt = text_prompt.replace(self.image_tokens,
                                          self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.generation_model.num_image_token + self.IMG_END_TOKEN,
                                          1)
        image = self.load_image(image_input)
        inputs = self.processor(text=text_prompt, images=[image], padding=True, return_tensors='pt').to(
                self.generation_model.device)

        input_ids = inputs['input_ids']
        inputs_embeds = self.generation_model.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.generation_model.extract_feature(inputs['pixel_values'])
        B, N, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.processor.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN))
        assert selected.sum() != 0
        inputs_embeds[selected] = vit_embeds.reshape(-1, C).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.reshape(B, N, C)

        return inputs_embeds

    @override
    def generate(self, **inputs):
        self.generation_model.img_context_token_id = self.processor.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        return self.generation_model.generate(**inputs)
