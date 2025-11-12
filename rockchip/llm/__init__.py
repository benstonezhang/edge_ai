def get_model_path(model_name: str):
    import os
    from huggingface_hub.utils import scan_cache_dir

    for _ in scan_cache_dir().repos:
        if _.repo_id == model_name:
            return os.path.join(_.repo_path, 'snapshots', _.refs['main'].commit_hash)
    print('Model path search failed')
    exit(1)


# 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定
# 若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
def from_pretrained(model_name: str, device_map='auto', low_cpu_mem_usage=True, trust_remote_code=True, **kwargs):
    _, name = model_name.split('/')
    name = name.lower()
    if name == 'gemma-3' or name[:8] == 'gemma-3-':
        from .gemma3 import Gemma3MultiModalModel as VisionModel
    elif name[:8] == 'gemma-3n':
        from .gemma3n import Gemma3nMultiModalModel as VisionModel
    elif name[:8] == 'qwen2-vl':
        if kwargs['height'] != 448 or kwargs['width'] != 448:
            raise RuntimeError('Qwen input image must be 448x448')
        from .qwen2_vl import Qwen2_VLMultiModalModel as VisionModel
    elif name[:10] == 'qwen2.5-vl':
        if kwargs['height'] != 448 or kwargs['width'] != 448:
            raise RuntimeError('Qwen input image must be 448x448')
        from .qwen2_5_vl import Qwen2_5_VLMultiModalModel as VisionModel
    elif name[:13] == 'minicpm-v-2_6':
        from .minicpm_v_2_6 import MiniCPMV2_6MultiModalModel as VisionModel
    elif name[:7] == 'smolvlm':
        from .smolvlm import SmolVLMMultiModalModel as VisionModel
    elif name[:9] == 'internvl3':
        from .internvl3 import InternVL3MultiModalModel as VisionModel
    else:
        print(f'unsupported model: {model_name}')
        exit(1)

    return VisionModel.from_pretrained(model_name, device_map=device_map, low_cpu_mem_usage=low_cpu_mem_usage,
                                       trust_remote_code=trust_remote_code, **kwargs)
