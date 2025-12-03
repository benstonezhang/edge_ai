import torch
from typing_extensions import override

from .qwen_vl import Qwen_VLMultiModalModel, Qwen_VLVisionForOnnx


class Qwen2_5_VLVisionForOnnx(Qwen_VLVisionForOnnx):
    def __init__(self, *args, **kwargs):
        import numpy as np

        super().__init__(*args, **kwargs)
        grid_thw = np.array([[self.grid_t, self.grid_h, self.grid_w]])

        def forward(pixel_values, **kwargs):
            hidden_states = self.vpm.patch_embed(pixel_values)
            rotary_pos_emb = self.vpm.rot_pos_emb(grid_thw)
            window_index, cu_window_seqlens = self.vpm.get_window_index(grid_thw)
            cu_window_seqlens = torch.tensor(cu_window_seqlens, device=hidden_states.device, dtype=torch.int32)
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

            seq_len, _ = hidden_states.size()
            hidden_states = hidden_states.reshape(seq_len // self.vpm.spatial_merge_unit,
                                                  self.vpm.spatial_merge_unit, -1)
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.vpm.spatial_merge_unit,
                                                    self.vpm.spatial_merge_unit, -1)
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            cu_seqlens = torch.repeat_interleave(torch.asarray(self.grid_h * self.grid_w),
                                                 torch.asarray(self.grid_t)).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

            for layer_num, blk in enumerate(self.vpm.blocks):
                hidden_states = blk(hidden_states,
                                    cu_seqlens=cu_seqlens if layer_num in self.vpm.fullatt_block_indexes else cu_window_seqlens,
                                    position_embeddings=position_embeddings, **kwargs)

            hidden_states = self.vpm.merger(hidden_states)
            reverse_indices = torch.argsort(window_index)
            hidden_states = hidden_states[reverse_indices, :]

            return hidden_states

        self.vpm.forward = forward


class Qwen2_5_VLMultiModalModel(Qwen_VLMultiModalModel):
    processor_config = {**Qwen_VLMultiModalModel.processor_config, "use_fast": False}

    @override
    @classmethod
    def from_pretrained(cls, model_name: str, load_processor: bool, *args, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, *args, _attn_implementation="eager", **kwargs)
        self.model = self.generation_model.model
        self.vision_model = self.generation_model.visual if hasattr(self.generation_model, 'visual') \
            else self.model.visual
        self.embed_tokens = self.model.embed_tokens if hasattr(self.model, 'embed_tokens') \
            else self.model.language_model.embed_tokens
        if load_processor:
            from transformers import Qwen2_5_VLProcessor

            self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name, **self.processor_config)
        return self

    @override
    def get_vision(self):
        return Qwen2_5_VLVisionForOnnx(self.vision_model, 1, self.image_size, self.image_size)

    # @staticmethod
    # def get_vision_forward_params(batch_size):
    #     grid_thw = torch.tensor([[(batch_size + Qwen2_5_VLVisionForOnnx.temporal_patch_size - 1
    #                                ) // Qwen2_5_VLVisionForOnnx.temporal_patch_size,
    #                               Qwen2_5_VLMultiModalModel.image_size // Qwen2_5_VLVisionForOnnx.patch_size,
    #                               Qwen2_5_VLMultiModalModel.image_size // Qwen2_5_VLVisionForOnnx.patch_size]],
    #                             dtype=torch.int64)
    #     onnx_export_config = {
    #         'input_names': [*MultiModalModel.onnx_input_names, 'grid_thw'],
    #         'output_names': MultiModalModel.onnx_output_names,
    #         # 'dynamic_axes': {'pixel': {2: 'height', 3: 'width'}},
    #         # 'dynamic_shapes': {'pixel': {2: 'height', 3: 'width'}},
    #     }
    #     return (grid_thw,), onnx_export_config

    # @staticmethod
    # def get_rknn_config(batch_size):
    #     return {
    #         'inputs': [*MultiModalModel.onnx_input_names, 'grid_thw'],
    #         'input_size_list': [
    #             [batch_size, 3, Qwen2_5_VLMultiModalModel.image_size, Qwen2_5_VLMultiModalModel.image_size],
    #             [batch_size, 3]
    #         ]
    #     }
