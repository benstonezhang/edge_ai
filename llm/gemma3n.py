import torch
from typing_extensions import override

from .base import MultiModalModel


class Gemma3nVisionForOnnx(torch.nn.Module):
    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm

    def forward(self, pixel_values):
        return self.vlm.get_image_features(pixel_values)


class Gemma3nMultiModalModel(MultiModalModel):
    vision_mean = [123.675, 116.28, 103.53]
    vision_std = [58.395, 58.395, 58.395]
    image_size = (768, 768)
    image_token = '<image_soft_token>'
    generation_config = {'max_new_tokens': 1024, 'do_sample': False}
    output_prefix_with_prompt = True

    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        from transformers import Gemma3nForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = Gemma3nForConditionalGeneration.from_pretrained(*args, **kwargs)
        self.model = self.generation_model.model
        return self

    @override
    def get_vision(self):
        return Gemma3nVisionForOnnx(self.model)

    @override
    def get_input_embeddings(self, processor, text_input: str, image_input: str,
                             audio_input=None, audio_input_mask=None):
        from PIL import Image

        image = Image.open(image_input)
        inputs = processor(text=['<image_soft_token> ' + text_input],
                           images=[image],
                           padding=True,
                           return_tensors='pt').to(self.model.device)
        input_ids = inputs['input_ids']
        pixel_values = inputs['pixel_values']

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Handle vision tokens (>= embed_vision.vocab_offset and < embed_audio.vocab_offset)
        vision_mask = torch.logical_and(input_ids >= self.model.embed_vision.vocab_offset,
                                        input_ids < self.model.embed_audio.vocab_offset)
        dummy_vision_token_id = self.model.embed_vision.vocab_offset + self.model.embed_vision.vocab_size - 1
        vision_input_ids = torch.where(vision_mask, input_ids, dummy_vision_token_id).to(inputs_embeds.device)
        vision_embeds = self.model.embed_vision(input_ids=vision_input_ids)
        expanded_vision_mask = vision_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(expanded_vision_mask, vision_embeds, inputs_embeds)

        # Handle audio tokens (>= embed_audio.vocab_offset)
        audio_mask = input_ids >= self.model.embed_audio.vocab_offset
        dummy_audio_token_id = self.model.embed_audio.vocab_offset + self.model.embed_audio.vocab_size - 1
        audio_input_ids = torch.where(audio_mask, input_ids, dummy_audio_token_id).to(inputs_embeds.device)
        audio_embeds = self.model.embed_audio(input_ids=audio_input_ids)
        expanded_audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(expanded_audio_mask, audio_embeds, inputs_embeds)

        image_features = self.model.get_image_features(pixel_values).to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask, _ = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                                image_features=image_features)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if audio_input is not None and audio_input_mask is not None:
            audio_features, audio_mask = self.model.get_audio_features(audio_input, ~audio_input_mask)

            # The Gemma3nProcessor expects all audio will be 30s in length and inserts 188 audio soft tokens into the
            # text to account for this. However, the audio preprocessing and encoder do not gurarantee they will
            # produce 188 soft tokens; they will produce at most that many tokens, but they may produce fewer tokens
            # depending on the length of the longest audio input in the batch. When we encounter this situation, we pad
            # the audio feature out to 188 soft tokens with the emebedding of the last token in the embed_audio vocab.
            audio_padding_tokens = torch.tensor([[self.model.vocab_size - 1]], dtype=torch.long,
                                                device=audio_features.device)
            audio_padding_embs = self.model.embed_audio(input_ids=audio_padding_tokens)
            audio_features = torch.where(audio_mask.unsqueeze(-1), audio_padding_embs, audio_features)

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
            extra_padding_tokens = self.model.config.audio_soft_tokens_per_image - audio_seq_len
            extra_padding_features = audio_padding_embs.expand(audio_batch_size, extra_padding_tokens, audio_embed_dim)

            audio_features = torch.cat((audio_features, extra_padding_features), dim=1)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, special_audio_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                                    audio_features=audio_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        return inputs_embeds
