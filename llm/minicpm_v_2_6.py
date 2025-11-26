import torch
from typing_extensions import override

from .model import MultiModalModel


class MiniCPMV2_6VisionForOnnx(torch.nn.Module):
    def __init__(self, vlm, batch_size, in_h, in_w):
        import math

        super().__init__()
        self.vpm = vlm.vpm
        self.resampler = vlm.resampler
        patch_size = vlm.config.patch_size
        num_patches_per_side = vlm.vpm.embeddings.num_patches_per_side
        tgt_sizes = torch.Tensor([[(in_h // patch_size), math.ceil(in_w / patch_size)]]).type(torch.int32)
        patch_attention_mask = torch.ones(size=(batch_size, in_h // patch_size, in_w // patch_size),
                                          dtype=torch.bool, device=vlm.device)
        max_im_h, max_im_w = in_h, in_w
        max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
            position_ids = position_ids.to(vlm.device)

        self.position_ids = position_ids

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((batch_size, max_patch_len), dtype=torch.bool, device=vlm.device)
        pos_embed = []
        for i in range(batch_size):
            tgt_h, tgt_w = tgt_sizes[i]
            # patches * D
            pos_embed.append(self.resampler.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(torch.float32))
            key_padding_mask[i, patch_len[i]:] = True

        self.pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0
                                                         ).permute(1, 0, 2)  # BLD => L * B * D

    def forward(self, pixel_values):
        batch_size = pixel_values.size(0)
        # patch embedding
        patch_embeds = self.vpm.embeddings.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        hidden_states = embeddings + self.vpm.embeddings.position_embedding(self.position_ids)
        # encoder
        encoder_outputs = self.vpm.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vpm.post_layernorm(last_hidden_state)
        # resampler
        x = self.resampler.kv_proj(last_hidden_state)  # B * L * D
        x = self.resampler.ln_kv(x).permute(1, 0, 2)  # L * B * D
        q = self.resampler.ln_q(self.resampler.query)  # Q * D

        out = self.resampler.attn(self.resampler._repeat(q, batch_size),  # Q * B * D
                                  x + self.pos_embed,  # L * B * D +  L * B * D
                                  x)[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.resampler.ln_post(x)
        x = x @ self.resampler.proj
        return x


class MiniCPMV2_6MultiModalModel(MultiModalModel):
    image_size = 448
    image_token = '(<image>./</image>)'
    processor_config = {"use_fast": False}

    @override
    @classmethod
    def from_pretrained(cls, model_name: str, load_processor: bool, *args, **kwargs):
        from transformers import AutoModel

        self = cls.__new__(cls)
        self.generation_model = AutoModel.from_pretrained(model_name, *args, trust_remote_code=True, **kwargs)
        self.model = self.generation_model
        if load_processor:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **self.processor_config)
        return self

    @override
    def get_vision(self):
        return MiniCPMV2_6VisionForOnnx(self.model, 1, self.image_size, self.image_size)

    @override
    def get_input_embeddings(self, text_input: str, image_input: str, audio_input=None, audio_input_mask=None):
        from PIL import Image

        image = Image.open(image_input)
        inputs = self.processor(text=self.image_token + text_input,
                                images=[image],
                                padding=True,
                                return_tensors='pt').to(self.model.device)
        inputs_embeds = self.model.get_vllm_embedding(inputs)[0]

        return inputs_embeds

    @override
    def generate(self, **inputs):
        del inputs['image_sizes']
        return self.generation_model.generate(**inputs, tokenizer=self.processor.tokenizer)
