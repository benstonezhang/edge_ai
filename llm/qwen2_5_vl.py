import torch
from typing_extensions import override

from .model import MultiModalModel


class Qwen2_5_VLVisionForOnnx(torch.nn.Module):
    merge_size = 2
    temporal_patch_size = 2
    patch_size = 14
    channel = 3

    def __init__(self, vpm, batch_size, height, width):
        super().__init__()
        self.vpm = vpm
        self.batch_size = batch_size
        self.grid_t = (batch_size + self.temporal_patch_size - 1) // self.temporal_patch_size
        self.grid_h = height // self.patch_size
        self.grid_w = width // self.patch_size

        # grid_thw = torch.tensor([[self.grid_t, self.grid_h, self.grid_w]], device=vpm.device, dtype=torch.int64)
        # rotary_pos_emb = vpm.rot_pos_emb(grid_thw).cpu().detach().numpy()
        # window_index, cu_window_seqlens = vpm.get_window_index(grid_thw)
        # window_index = window_index.cpu().detach().numpy()
        # cu_window_seqlens = torch.tensor(cu_window_seqlens, device=grid_thw.device, dtype=torch.int32)
        # cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens).cpu().detach().numpy()
        # cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        #                                      ).cumsum(dim=0, dtype=torch.int32)
        # cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0).cpu().detach().numpy()
        #
        # def forward(hidden_states):
        #     hidden_states = vpm.patch_embed(hidden_states)
        #
        #     emb = torch.from_numpy(rotary_pos_emb).to(dtype=hidden_states.dtype, device=hidden_states.device)
        #     w_idx = torch.from_numpy(window_index).to(dtype=torch.int32, device=hidden_states.device)
        #     w_seqlens = torch.from_numpy(cu_window_seqlens).to(dtype=torch.int32, device=hidden_states.device)
        #     seqlens = torch.from_numpy(cu_seqlens).to(dtype=torch.int32, device=hidden_states.device)
        #
        #     seq_len, _ = hidden_states.size()
        #     hidden_states = hidden_states.reshape(seq_len // vpm.spatial_merge_unit, vpm.spatial_merge_unit, -1)
        #     hidden_states = hidden_states[w_idx, :, :]
        #     hidden_states = hidden_states.reshape(seq_len, -1)
        #     emb = emb.reshape(seq_len // vpm.spatial_merge_unit, vpm.spatial_merge_unit, -1)
        #     emb = emb[w_idx, :, :]
        #     emb = emb.reshape(seq_len, -1)
        #     emb = torch.cat((emb, emb), dim=-1)
        #     position_embeddings = (emb.cos(), emb.sin())
        #
        #     for layer_num, blk in enumerate(vpm.blocks):
        #         hidden_states = blk(hidden_states,
        #                             cu_seqlens=seqlens if layer_num in vpm.fullatt_block_indexes else w_seqlens,
        #                             position_embeddings=position_embeddings)
        #
        #     hidden_states = vpm.merger(hidden_states)
        #     reverse_indices = torch.argsort(w_idx)
        #     hidden_states = hidden_states[reverse_indices, :]
        #
        #     return hidden_states
        #
        # self.vpm.forward = forward

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
        grid_thw = torch.tensor([[self.grid_t, self.grid_h, self.grid_w]], device=self.vpm.device, dtype=torch.int64)
        patches = pixel_inputs.reshape(self.grid_t, self.temporal_patch_size, self.channel,
                                       self.grid_h // self.merge_size, self.merge_size, self.patch_size,
                                       self.grid_w // self.merge_size, self.merge_size, self.patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(self.grid_t * self.grid_h * self.grid_w,
                                          self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)

        return self.vpm(flatten_patches, grid_thw=grid_thw)


# The default range for the number of visual tokens per image in the model is 4-16384.You can set min_pixels and
# max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

class Qwen2_5_VLMultiModalModel(MultiModalModel):
    vision_mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    vision_std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
    image_size = 448
    # image_tokens = '<|vision_start|><|image_pad|>...<|vision_end|>'
    processor_config = {"use_fast": False, "min_pixels": 256 * 28 * 28, "max_pixels": 2048 * 28 * 28}

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
    #         'input_names': ['pixel'],
    #         'dynamic_axes': {'pixel': {2: 'height', 3: 'width'}},
    #         # 'dynamic_shapes': {'pixel': {2: 'height', 3: 'width'}},
    #     }
    #     return (grid_thw,), onnx_export_config
    #
    # @staticmethod
    # def get_onnx_load_config(batch_size):
    #     return {
    #         'inputs': [*MultiModalModel.onnx_input_names, 'grid_thw'],
    #         'input_size_list': [
    #             [batch_size, 3, Qwen2_5_VLMultiModalModel.image_size, Qwen2_5_VLMultiModalModel.image_size],
    #             [batch_size, 3]
    #         ]
    #     }

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

        # Preprocess the inputs
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
        image = Image.open(image_input)
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(
                self.model.device)

        inputs_embeds = self.embed_tokens(inputs["input_ids"])
        pixel_values = inputs["pixel_values"].type(self.vision_model.dtype)
        image_mask = inputs["input_ids"] == self.model.config.image_token_id
        image_embeds = self.vision_model(pixel_values, grid_thw=inputs["image_grid_thw"]).to(inputs_embeds.device)
        inputs_embeds[image_mask] = image_embeds

        return inputs_embeds
