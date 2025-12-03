import torch
from typing_extensions import override

from .qwen_vl import Qwen_VLMultiModalModel, Qwen_VLVisionForOnnx


class Qwen2_VLVisionForOnnx(Qwen_VLVisionForOnnx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        grid_thw = torch.tensor([[self.grid_t, self.grid_h, self.grid_w]], dtype=torch.int64)
        rotary_pos_emb = self.vpm.rot_pos_emb(grid_thw).cpu().detach().numpy()
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).cpu().detach().numpy()

        def forward(hidden_states):
            hidden_states = self.vpm.patch_embed(hidden_states)
            emb = torch.from_numpy(rotary_pos_emb).to(dtype=hidden_states.dtype, device=hidden_states.device)
            emb = torch.cat((emb, emb), dim=-1)
            seqlens = torch.from_numpy(cu_seqlens).to(dtype=torch.int32, device=hidden_states.device)
            for blk in self.vpm.blocks:
                hidden_states = blk(hidden_states, cu_seqlens=seqlens,
                                    position_embeddings=(emb.cos(), emb.sin()))
            return self.vpm.merger(hidden_states)

        self.vpm.forward = forward


class Qwen2_VLMultiModalModel(Qwen_VLMultiModalModel):
    processor_config = {**Qwen_VLMultiModalModel.processor_config, "use_fast": False}

    @override
    @classmethod
    def from_pretrained(cls, model_name: str, load_processor: bool, *args, **kwargs):
        from transformers import Qwen2VLForConditionalGeneration

        self = cls.__new__(cls)
        self.generation_model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, *args, **kwargs)
        self.model = self.generation_model.model
        self.vision_model = self.generation_model.visual if hasattr(self.generation_model, 'visual') \
            else self.model.visual
        self.embed_tokens = self.model.embed_tokens if hasattr(self.model, 'embed_tokens') \
            else self.model.language_model.embed_tokens
        if load_processor:
            from transformers import Qwen2VLProcessor

            self.processor = Qwen2VLProcessor.from_pretrained(model_name, **self.processor_config)
        return self

    @override
    def get_vision(self):
        return Qwen2_VLVisionForOnnx(self.vision_model, 1, self.image_size, self.image_size)
