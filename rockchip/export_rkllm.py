import argparse
import json
import os


def check_json_valid(f: str):
    if os.path.exists(f):
        with open(f, 'r') as json_file:
            try:
                inputs = json.load(json_file)
                if isinstance(inputs, list) and len(inputs) > 0:
                    return True
            except Exception:
                pass
    return False


def generate_tokens(model_name: str, data_dir: str, inputs_json: str):
    from tqdm import tqdm

    from llm.utils import from_pretrained, get_device_and_dtype

    device_map, torch_dtype = get_device_and_dtype()
    model = from_pretrained(args.model_name, device_map=device_map, torch_dtype=torch_dtype).eval()
    datasets = json.load(open(os.path.join(data_dir, 'datasets.json'), 'r'))
    first_line = True
    with open(inputs_json, 'w') as json_file:
        json_file.write('[\n')
        for data in tqdm(datasets):
            image_file = os.path.join(data_dir, 'datasets', data['image'])
            inputs_embeds = model.get_input_embeddings(text_input=data['input'], image_input=image_file)

            if first_line:
                first_line = False
            else:
                json_file.write(',\n')
            json.dump({
                'input_embed': inputs_embeds.tolist(),
                'target': data['target'],
            }, json_file)
        json_file.write('\n]')


def main(args: argparse.Namespace):
    if args.target_platform == 'rk3588':
        num_npu_core = 3
        quantized_dtype = 'w8a8'
        quantized_algorithm = 'normal'
    elif args.target_platform == 'rk3576':
        num_npu_core = 2
        quantized_dtype = 'w4a16'
        quantized_algorithm = 'grq'
    elif args.target_platform in ['rk3562', 'rk3566', 'rk3568', 'rk2118', 'rv1106', 'rv1103', 'rv1126b']:
        num_npu_core = 1
        quantized_dtype = 'w4a16_g128'
        quantized_algorithm = 'grq'
    else:
        print(f'unsupported platform: {args.target_platform}')
        exit(1)

    optimization_level = 1

    data_dir = os.path.realpath(args.data_dir)
    os.makedirs(args.out_dir, mode=0o755, exist_ok=True)
    os.chdir(args.out_dir)

    model_name = args.model_name.replace('/', '-').lower()
    inputs_json = f'{model_name}_inputs.json'
    model_rkllm = f'{model_name}_{args.target_platform}_{quantized_dtype}_{quantized_algorithm}.rkllm'

    if not check_json_valid(inputs_json):
        generate_tokens(model_name=args.model_name, data_dir=data_dir, inputs_json=inputs_json)

    if not os.path.exists(model_rkllm):
        from llm.utils import get_model_path
        from rkllm.api import RKLLM

        llm = RKLLM()
        model_path = get_model_path(args.model_name)
        print(f'Loading model {args.model_name} from {model_path}')
        ret = llm.load_huggingface(model=model_path,
                                   model_lora=None,
                                   device='cpu',
                                   dtype='float32',
                                   custom_config=None,
                                   load_weight=True)
        # ret = llm.load_gguf(model = args.model_name)
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        ret = llm.build(do_quantization=True,
                        optimization_level=optimization_level,
                        quantized_dtype=quantized_dtype,
                        quantized_algorithm=quantized_algorithm,
                        target_platform=args.target_platform,
                        num_npu_core=num_npu_core,
                        extra_qparams=None,
                        dataset=inputs_json,
                        hybrid_rate=0,
                        max_context=4096)
        if ret != 0:
            print('Build model failed!')
            exit(ret)

        ret = llm.export_rkllm(model_rkllm)
        if ret != 0:
            print('Export model failed!')
            exit(ret)

    print('Done')


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_name', type=str, default=None, help='model name', required=True)
    argparse.add_argument('--data_dir', type=str, default='data', help='data folder', required=False)
    argparse.add_argument('--out_dir', type=str, default='out', help='output folder', required=False)
    argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
    argparse.add_argument('--target_platform', type=str, default='rk3576',
                          help='target platform, choose from [rk3588, rk3576, rk3562, rk3566, rk3568, rk2118, rv1106, rv1103, rv1126b]',
                          required=False)
    args = argparse.parse_args()

    main(args)
