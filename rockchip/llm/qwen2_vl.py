import os.path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing_extensions import override

from .base import MultiModalModel


class Qwen2_VLVisionForOnnx(torch.nn.Module):
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

        def forward(hidden_states, grid_thw=None):
            hidden_states = self.vpm.patch_embed(hidden_states)
            if grid_thw is not None:
                rotary_pos_emb = self.vpm.rot_pos_emb(grid_thw)
                cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                     grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                np.save("rotary_pos_emb.npy", rotary_pos_emb.cpu().detach().numpy())
                np.save("cu_seqlens.npy", cu_seqlens.cpu().detach().numpy())
            else:
                rotary_pos_emb = torch.from_numpy(np.load("rotary_pos_emb.npy")).to(
                    dtype=hidden_states.dtype, device=hidden_states.device)
                cu_seqlens = torch.from_numpy(np.load("cu_seqlens.npy")).to(
                    dtype=torch.int32, device=hidden_states.device)

            for blk in self.vpm.blocks:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

            return self.vpm.merger(hidden_states)

        self.vpm.forward = forward

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

        if not (os.path.exists('rotary_pos_emb.npy') and os.path.exists('cu_seqlens.npy')):
            def forward(hidden_states, grid_thw=None):
                hidden_states = self.vpm.patch_embed(hidden_states)
                rotary_pos_emb = self.vpm.rot_pos_emb(grid_thw)
                cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                     grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                np.save("rotary_pos_emb.npy", rotary_pos_emb.cpu().detach().numpy())
                np.save("cu_seqlens.npy", cu_seqlens.cpu().detach().numpy())

                for blk in self.vpm.blocks:
                    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

                return self.vpm.merger(hidden_states)

            self.vpm.forward = forward
            grid_t = self.batch_size // self.temporal_patch_size if self.batch_size % self.temporal_patch_size == 0 else self.batch_size // self.temporal_patch_size + 1
            grid_h = self.height // self.patch_size
            grid_w = self.width // self.patch_size
            self.vpm(flatten_patches, torch.tensor([grid_t, grid_h, grid_w]).unsqueeze(0))

        def forward(hidden_states):
            hidden_states = self.vpm.patch_embed(hidden_states)
            rotary_pos_emb = torch.from_numpy(np.load("rotary_pos_emb.npy")).to(dtype=hidden_states.dtype,
                                                                                device=hidden_states.device)
            cu_seqlens = torch.from_numpy(np.load("cu_seqlens.npy")).to(dtype=torch.int32, device=hidden_states.device)
            for blk in self.vpm.blocks:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            return self.vpm.merger(hidden_states)

        self.vpm.forward = forward
        return self.vpm(flatten_patches)


class Qwen2_VLMultiModalModel(MultiModalModel):
    @override
    @classmethod
    def from_pretrained(cls, *args, batch_size=1, height=None, width=None, **kwargs):
        self = cls.__new__(cls)
        self.generation_model = Qwen2VLForConditionalGeneration.from_pretrained(*args, **kwargs)
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
        return Qwen2_VLVisionForOnnx(self.model, self.batch_size, self.height, self.width)

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
