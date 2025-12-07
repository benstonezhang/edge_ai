import typing


def rknn_export(onnx_model: str, target_platform: str, rknn_model: str,
                mean_values: list[float], std_values: list[float], rknn_onnx_config: typing.Dict, debug: bool):
    from rknn.api import RKNN

    rknn = RKNN(verbose=debug)
    rknn.config(target_platform=target_platform, mean_values=mean_values, std_values=std_values)
    rknn.load_onnx(onnx_model, **rknn_onnx_config)
    rknn.build(do_quantization=False, dataset=None, rknn_batch_size=1)
    rknn.export_rknn(rknn_model)


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='model name', required=False)
    parser.add_argument('--onnx_path', type=str, default=None, help='onnx model path', required=False)
    parser.add_argument('--out_dir', type=str, default='out', help='output folder', required=False)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
    parser.add_argument('--target_platform', type=str, default='rk3576',
                        help='target platform, choose from [rk3588, rk3576, rk3562, rk3566, rk3568, rk2118, rv1106, rv1103, rv1126b]',
                        required=False)
    parser.add_argument('--debug', action='store_true', help='debug code')
    parser.add_argument('--force', action='store_true', help='overwrite if output exist')
    return parser.parse_args()


def main():
    import os
    from llm.utils import bare_model

    args = arg_parser()
    if args.model_name is None and args.onnx_path is None:
        print('model_name or onnx_path is required')
        exit(1)

    if args.model_name is not None:
        model_name = args.model_name.replace('/', '-').lower()
        vision_onnx = f'{model_name}_vision.onnx'
        vision_rknn = f'{model_name}_vision_{args.target_platform}.rknn'
    else:
        vision_onnx = os.path.realpath(args.onnx_path)
        vision_rknn = f'{".".join(os.path.basename(vision_onnx).split(".")[:-1])}.rknn'

    if not os.path.exists(os.path.join(args.out_dir, vision_onnx)):
        print(f'input model file {vision_onnx} not accessible')
        exit(1)

    if args.force or not os.path.exists(os.path.join(args.out_dir, vision_rknn)):
        os.makedirs(args.out_dir, mode=0o755, exist_ok=True)
        os.chdir(args.out_dir)

        model = bare_model(args.model_name)
        vision_mean = model.vision_mean
        vision_std = model.vision_std
        rknn_onnx_config = model.get_rknn_config(args.batch_size) if hasattr(model, 'get_rknn_config') else {}
        del model

        rknn_export(onnx_model=vision_onnx, target_platform=args.target_platform, rknn_model=vision_rknn,
                    mean_values=vision_mean, std_values=vision_std, rknn_onnx_config=rknn_onnx_config, debug=args.debug)


if __name__ == '__main__':
    main()
