def get_model_path(model_name: str):
    import os
    from huggingface_hub.utils import scan_cache_dir

    for _ in scan_cache_dir().repos:
        if _.repo_id == model_name:
            return os.path.join(_.repo_path, 'snapshots', _.refs['main'].commit_hash)
    print('Model path search failed')
    exit(1)


def from_pretrained(model_name: str, *args, **kwargs):
    import torch

    if model_name[:15] == 'google/gemma-3n':
        from .gemma3n import Gemma3nMultiModalModel as VisionModel
    else:
        print(f'unsupported model: {model_name}')
        exit(1)

    return VisionModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True,
        trust_remote_code=True)
