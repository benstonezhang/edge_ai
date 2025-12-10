import torch


def onnx_show_shape(out):
    if torch.is_tensor(out):
        print('vision output:', out.shape)
    else:
        shapes = []
        for o in out:
            if torch.is_tensor(o):
                shapes.append(o.shape)
                continue
            shapes.append('[')
            for _ in o:
                shapes.append(_.shape)
            shapes.append(']')
        print('vision output:', *shapes)


def onnx_export(model_name: str, batch_size: int, img_height: int, img_width: int, model_file: str, opset: int):
    from llm.utils import from_pretrained

    model = from_pretrained(model_name, device_map='cpu', dtype=torch.float32).eval()
    vision_model = model.get_vision().eval()

    pixel_values = torch.randn(batch_size, 3, img_height, img_width, dtype=model.model.dtype)
    print('pixel_values:', pixel_values.shape)

    if hasattr(model, 'get_vision_forward_params'):
        forward_args, export_conf = model.get_vision_forward_params(batch_size)
    else:
        forward_args = tuple()
        export_conf = model.onnx_export_conf

    out = vision_model(pixel_values, *forward_args)
    onnx_show_shape(out)
    if model.dynamo_compatible:
        torch.onnx.export(vision_model, (pixel_values, *forward_args), model_file, opset_version=opset,
                          dynamo=True, optimize=True, fallback=True, **export_conf)
    else:
        torch.onnx.export(vision_model, (pixel_values, *forward_args), model_file, opset_version=opset,
                          dynamo=False, **export_conf)


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
    onnx_show_shape(out)


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='model name', required=False)
    parser.add_argument('--opset', type=int, default=19, help='onnx opset', required=False)
    parser.add_argument('--data_dir', type=str, default='data', help='data folder', required=False)
    parser.add_argument('--out_dir', type=str, default='out', help='output folder', required=False)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
    parser.add_argument('--show', action='store_true', help='show onnx model nodes')
    parser.add_argument('--verify', action='store_true', help='verify onnx model')
    parser.add_argument('--force', action='store_true', help='overwrite if output exist')
    return parser.parse_args()


def main():
    import os
    from llm.utils import bare_model

    args = arg_parser()
    if args.model_name is None:
        print('model_name is required')
        exit(1)

    model_name = args.model_name.replace('/', '-').lower()
    vision_onnx = f'{model_name}_vision.onnx'
    data_dir = os.path.realpath(args.data_dir)

    os.makedirs(args.out_dir, mode=0o755, exist_ok=True)
    os.chdir(args.out_dir)

    model = bare_model(args.model_name)
    img_height, img_width = model.get_image_height_and_width()
    del model

    if args.force or not os.path.exists(vision_onnx):
        onnx_export(model_name=args.model_name, batch_size=args.batch_size, img_height=img_height,
                    img_width=img_width, model_file=vision_onnx, opset=args.opset)

    if args.show:
        onnx_show(vision_onnx)

    if args.verify:
        demo_image = os.path.join(data_dir, 'demo.jpg')
        onnx_verify(model_file=vision_onnx, img_height=img_height, img_width=img_width, image_path=demo_image)


if __name__ == '__main__':
    main()
