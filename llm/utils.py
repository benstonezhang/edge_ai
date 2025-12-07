def get_model_path(model_name: str):
    import os
    from huggingface_hub.utils import scan_cache_dir

    for _ in scan_cache_dir().repos:
        if _.repo_id == model_name:
            return os.path.join(_.repo_path, 'snapshots', _.refs['main'].commit_hash)
    print('Model path search failed')
    exit(1)


def _get_model_cls(model_name: str):
    _, name = model_name.split('/')
    name = name.lower()
    if name == 'gemma-3' or name[:8] == 'gemma-3-':
        from .gemma3 import Gemma3MultiModalModel as model
    elif name[:8] == 'gemma-3n':
        from .gemma3n import Gemma3nMultiModalModel as model
    elif name[:8] == 'qwen2-vl':
        from .qwen2_vl import Qwen2_VLMultiModalModel as model
    elif name[:10] == 'qwen2.5-vl':
        from .qwen2_5_vl import Qwen2_5_VLMultiModalModel as model
    elif name[:8] == 'qwen3-vl':
        from .qwen3_vl import Qwen3_VLMultiModalModel as model
    elif name[:13] == 'minicpm-v-2_6':
        from .minicpm_v_2_6 import MiniCPMV2_6MultiModalModel as model
    elif name[:7] == 'smolvlm':
        from .smolvlm import SmolVLMMultiModalModel as model
    elif name[:9] == 'internvl3':
        from .internvl import InternVLMultiModalModel as model
    else:
        print(f'unsupported model: {model_name}')
        exit(1)

    return model


def bare_model(model_name: str):
    return _get_model_cls(model_name)()


# 注意此处的数据类型，由于 rknn 目前仅支持 float32 ，因此需要指定
# 若是在加载权重时限制了数据类型，需要自行修改config.json中的 "use_flash_attn" 参数为 false
def from_pretrained(model_name: str,
                    load_processor: bool = True,
                    device_map: str = 'auto',
                    low_cpu_mem_usage: bool = True,
                    **kwargs):
    import torch

    if torch.__version__ < '2.6':
        kwargs['torch_dtype'] = kwargs['dtype']
        del kwargs['dtype']

    return _get_model_cls(model_name).from_pretrained(
            model_name,
            load_processor,
            device_map=device_map,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **kwargs)


def get_device_and_dtype():
    import torch

    device_map = 'cpu'
    torch_dtype = torch.float32
    if torch.cuda.is_available():
        device_map = 'cuda'
        torch_dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device_map = 'mps'
        torch_dtype = torch.bfloat16

    return device_map, torch_dtype
