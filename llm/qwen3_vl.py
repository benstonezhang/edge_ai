import torch
from typing_extensions import override

from .qwen_vl import Qwen_VLMultiModalModel, Qwen_VLVisionForOnnx


class Qwen3_VLVisionForOnnx(Qwen_VLVisionForOnnx):
    patch_size = 16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import numpy as np

        grid_thw = np.array([[self.grid_t, self.grid_h, self.grid_w]])

        def rot_pos_emb() -> torch.Tensor:
            merge_size = self.vpm.spatial_merge_size

            # max_hw = int(grid_thw[:, 1:].max().item())
            max_hw = max(self.grid_t, self.grid_h, self.grid_w)
            freq_table = self.vpm.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
            device = freq_table.device

            # total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
            total_tokens = self.grid_t * self.grid_h * self.grid_w
            pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

            offset = 0
            for num_frames, height, width in grid_thw:
                merged_h, merged_w = height // merge_size, width // merge_size

                block_rows = torch.arange(merged_h, device=device)  # block row indices
                block_cols = torch.arange(merged_w, device=device)  # block col indices
                intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
                intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

                # Compute full-resolution positions
                row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
                col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

                row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
                col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

                coords = torch.stack((row_idx, col_idx), dim=-1)

                if num_frames > 1:
                    coords = coords.repeat(num_frames, 1)

                num_tokens = coords.shape[0]
                pos_ids[offset: offset + num_tokens] = coords
                offset += num_tokens

            embeddings = freq_table[pos_ids]  # lookup rotary embeddings
            embeddings = embeddings.flatten(1)
            return embeddings

        def forward(pixel_values):
            hidden_states = self.vpm.patch_embed(pixel_values)

            pos_embeds = self.vpm.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds

            rotary_pos_emb = self.vpm.rot_pos_emb()

            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            cu_seqlens = torch.repeat_interleave(torch.asarray(grid_thw[:, 1] * grid_thw[:, 2]),
                                                 torch.asarray(grid_thw[:, 0])).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

            for layer_num, blk in enumerate(self.vpm.blocks):
                hidden_states = blk(hidden_states,
                                    cu_seqlens=cu_seqlens,
                                    position_embeddings=position_embeddings,
                                    **kwargs)

            return self.vpm.merger(hidden_states)

        self.vpm.rot_pos_emb = rot_pos_emb
        self.vpm.forward = forward

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


class Qwen3_VLMultiModalModel(Qwen_VLMultiModalModel):
    processor_config = {**Qwen_VLMultiModalModel.processor_config, "use_fast": False}
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
