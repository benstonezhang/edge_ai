import os


def rkllm_export(model_name: str, model_rkllm: str, **build_config):
    from llm.utils import get_model_path
    from rkllm.api import RKLLM

    llm = RKLLM()
    model_path = get_model_path(model_name)
    print(f'Loading model {model_name} from {model_path}')
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

    ret = llm.build(**build_config)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    ret = llm.export_rkllm(model_rkllm)
    if ret != 0:
        print('Export model failed!')
        exit(ret)


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='model name', required=True)
    parser.add_argument('--out_dir', type=str, default='out', help='output folder', required=False)
    parser.add_argument('--target_platform', type=str, default='rk3576',
                        help='target platform, choose from [rk3588, rk3576, rk3562, rk3566, rk3568, rk2118, rv1106, rv1103, rv1126b]',
                        required=False)
    parser.add_argument('--no_quantization', action='store_true', help='disable quantization', required=False)
    parser.add_argument('--quantized_dtype', type=str, default=None, help='force quantized dtype', required=False)
    parser.add_argument('--optimization_level', type=int, default=1, help='optimization level', required=False)
    parser.add_argument('--force', action='store_true', help='overwrite if output exist')
    return parser.parse_args()


def main():
    args = arg_parser()

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

    if args.quantized_dtype is not None:
        quantized_dtype = args.quantized_dtype

    if quantized_dtype == 'w8a8':
        quantized_algorithm = 'normal'

    if args.no_quantization:
        quantized_dtype = 'f16'
        quantized_algorithm = None
        build_config = dict(do_quantization=False)
    else:
        build_config = dict(do_quantization=True,
                            quantized_dtype=quantized_dtype,
                            quantized_algorithm=quantized_algorithm)

    model_name = args.model_name.replace('/', '-').lower()
    inputs_json = f'{model_name}_inputs.json'
    model_rkllm = f'{model_name}_{args.target_platform}_{quantized_dtype}{f"_{quantized_algorithm}" if quantized_algorithm is not None else ""}.rkllm'

    if not os.path.exists(os.path.join(args.out_dir, inputs_json)):
        print(f'input embeds file {inputs_json} not accessible')
        exit(1)

    if args.force or not os.path.exists(os.path.join(args.out_dir, model_rkllm)):
        os.makedirs(args.out_dir, mode=0o755, exist_ok=True)
        os.chdir(args.out_dir)

        rkllm_export(model_name=args.model_name, model_rkllm=model_rkllm, **build_config,
                     optimization_level=args.optimization_level,
                     target_platform=args.target_platform,
                     num_npu_core=num_npu_core,
                     extra_qparams=None,
                     dataset=inputs_json,
                     hybrid_rate=0,
                     max_context=4096)


if __name__ == '__main__':
    main()
