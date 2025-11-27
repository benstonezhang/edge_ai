import argparse
import os
import typing

import torch


def onnx_export(model_name: str, batch_size: int, img_height: int, img_width: int, model_file: str, opset: int):
    from llm.utils import from_pretrained

    model = from_pretrained(model_name, device_map='cpu', torch_dtype=torch.float32).eval()
    vision_model = model.get_vision().eval()

    pixel_values = torch.randn(batch_size, 3, img_height, img_width, dtype=model.model.dtype)
    print('pixel_values:', pixel_values.shape)

    if hasattr(model, 'get_vision_forward_params'):
        forward_args, export_conf = model.get_vision_forward_params(batch_size)
    else:
        forward_args = tuple()
        export_conf = {'input_names': model.onnx_input_names, 'output_names': model.onnx_output_names}

    out = vision_model(pixel_values, *forward_args)
    print('vision output:', out.shape)
    torch.onnx.export(vision_model, (pixel_values, *forward_args), model_file, opset_version=opset, **export_conf)
    # torch.onnx.export(vision_model, (pixel_values, *forward_args), model_file, opset_version=opset,
    #                   dynamo=True, optimize=True, fallback=True, **export_conf)


def onnx_show(model_file: str):
    import onnx

    for item in onnx.load(model_file).graph.node:
        print(item.name)


def onnx_verify(model_file: str, img_height: int, img_width: int, image_path: str):
    import onnxruntime as ort
    from PIL import Image
    from torchvision import transforms

    orig_image = Image.open(image_path)
    if orig_image.size[0] > img_width or orig_image.size[1] > img_height:
        orig_image.thumbnail((img_width, img_height), reducing_gap=None)
    image = Image.new(orig_image.mode, (img_width, img_height))
    image.paste(orig_image)

    sess = ort.InferenceSession(model_file, providers=ort.get_available_providers())
    input_name = sess.get_inputs()[0].name
    img_tensor = transforms.PILToTensor()(image)
    out = sess.run(None, {input_name: [img_tensor]})[0]
    print('verify vision output:', out.shape)


def rknn_export(onnx_model: str, target_platform: str, rknn_model: str,
                mean_values: list[float], std_values: list[float], rknn_onnx_config: typing.Dict):
    from rknn.api import RKNN

    rknn = RKNN(verbose=args.debug)
    rknn.config(target_platform=target_platform, mean_values=mean_values, std_values=std_values)
    rknn.load_onnx(onnx_model, **rknn_onnx_config)
    rknn.build(do_quantization=False, dataset=None, rknn_batch_size=1)
    rknn.export_rknn(rknn_model)


def main(args: argparse.Namespace):
    from llm.utils import bare_model

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
    data_dir = os.path.join(args.data_dir)

    os.makedirs(args.out_dir, mode=0o755, exist_ok=True)
    os.chdir(args.out_dir)

    model = bare_model(args.model_name)
    vision_mean = model.vision_mean
    vision_std = model.vision_std
    img_height, img_width = model.get_image_height_and_width()
    rknn_onnx_config = model.get_rknn_config(args.batch_size) if hasattr(model, 'get_rknn_config') else {}
    del model

    if args.model_name is not None and not os.path.exists(vision_onnx):
        onnx_export(model_name=args.model_name, batch_size=args.batch_size, img_height=img_height,
                    img_width=img_width, model_file=vision_onnx, opset=args.opset)

    if args.show_onnx:
        onnx_show(vision_onnx)

    if args.verify_onnx:
        demo_image = os.path.join(data_dir, 'demo.jpg')
        onnx_verify(model_file=vision_onnx, img_height=img_height, img_width=img_width, image_path=demo_image)

    rknn_export(onnx_model=vision_onnx, target_platform=args.target_platform, rknn_model=vision_rknn,
                mean_values=vision_mean, std_values=vision_std, rknn_onnx_config=rknn_onnx_config)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_name', type=str, default=None, help='model name', required=False)
    argparse.add_argument('--onnx_path', type=str, default=None, help='onnx model path', required=False)
    argparse.add_argument('--verify', action='store_true', help='verify model by inference')
    argparse.add_argument('--opset', type=int, default=19, help='onnx opset', required=False)
    argparse.add_argument('--data_dir', type=str, default='data', help='data folder', required=False)
    argparse.add_argument('--out_dir', type=str, default='out', help='output folder', required=False)
    argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
    argparse.add_argument('--target_platform', type=str, default='rk3576',
                          help='target platform, choose from [rk3588, rk3576, rk3562, rk3566, rk3568, rk2118, rv1106, rv1103, rv1126b]',
                          required=False)
    argparse.add_argument('--show_onnx', action='store_true', help='show onnx model nodes')
    argparse.add_argument('--verify_onnx', action='store_true', help='verify onnx model')
    argparse.add_argument('--debug', action='store_true', help='debug code')
    args = argparse.parse_args()

    main(args)
