import argparse
import json
import os

import torch


def onnx_show(model_file: str):
    import onnx

    for item in onnx.load(model_file).graph.node:
        print(item.name)


def onnx_export(model, model_file: str, batch_size: int, img_height: int, img_width: int, opset: int):
    pixel_values = torch.randn(batch_size, 3, img_height, img_width, dtype=model.model.dtype)
    print('pixel_values:', pixel_values.shape)
    vision_model = model.get_vision().eval()
    out = vision_model(pixel_values)
    print('vision output:', out.shape)
    torch.onnx.export(vision_model, (pixel_values,), model_file, input_names=['pixel'], opset_version=opset)


def onnx_verify(model_file: str, img_height: int, img_width: int, image_path: str):
    from PIL import Image
    import onnxruntime as ort
    from torchvision import transforms

    sess = ort.InferenceSession(model_file, providers=ort.get_available_providers())
    input_name = sess.get_inputs()[0].name
    image = Image.open(image_path).resize((img_height, img_width))
    img_tensor = transforms.PILToTensor()(image)
    out = sess.run(None, {input_name: [img_tensor]})[0]
    print('verify vision output:', out.shape)


def rknn_export(onnx_model: str, target_platform: str, rknn_model: str, mean_values: list[float],
                std_values: list[float]):
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(target_platform=target_platform, mean_values=mean_values, std_values=std_values)
    rknn.load_onnx(onnx_model)
    rknn.build(do_quantization=False, rknn_batch_size=1)
    rknn.export_rknn(rknn_model)


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


def generate_tokens(model, model_name: str, data_path: str, inputs_json: str):
    from tqdm import tqdm
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    datasets = json.load(open(os.path.join(data_path, 'datasets.json'), 'r'))
    first_line = True
    with open(inputs_json, 'w') as json_file, tqdm(datasets) as bar:
        json_file.write('[\n')
        for data in datasets:
            image = Image.open(os.path.join(data_path, 'datasets', data['image']))
            inputs_embeds = model.get_input_embeddings(processor=processor, text_input=data['input'],
                                                       image_input=image)
            bar.write(f'inputs_embeds {inputs_embeds.shape}')

            if first_line:
                first_line = False
            else:
                json_file.write(',\n')
            json.dump({
                'input_embed': inputs_embeds.tolist(),
                'target': data['target'],
            }, json_file)

            bar.update()

        json_file.write('\n]')


def main(args: argparse.Namespace):
    from llm import from_pretrained

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

    img_height, img_width = (int(x) for x in args.image_size.split('x'))
    optimization_level = 1
    model_name = args.model_name.replace('/', '-')

    os.makedirs(args.out_path, mode=0o755, exist_ok=True)
    vision_onnx = os.path.join(args.out_path, f'{model_name}_vision.onnx')
    vision_rknn = os.path.join(args.out_path, f'{model_name}_{args.target_platform}_vision.rknn')
    inputs_json = os.path.join(args.out_path, f'{model_name}_inputs.json')
    model_rkllm = os.path.join(args.out_path,
                               f'{model_name}_{args.target_platform}_{quantized_dtype}_{quantized_algorithm}.rkllm')

    if not os.path.exists(vision_rknn):
        model = from_pretrained(args.model_name, device_map='cpu', torch_dtype=torch.float32,
                                batch_size=args.batch_size, height=img_height, width=img_width).eval()

        if not os.path.exists(vision_onnx):
            onnx_export(model, vision_onnx, args.batch_size, img_height, img_width, args.opset)

        vision_mean = model.vision_mean
        vision_std = model.vision_std
        del model

        if args.verify_onnx:
            onnx_verify(vision_onnx, img_height, img_width, os.path.join(args.data_path, 'demo.jpg'))

        if args.show_onnx:
            onnx_show(vision_onnx)

        rknn_export(vision_onnx, args.target_platform, vision_rknn, vision_mean, vision_std)

    if not check_json_valid(inputs_json):
        device_map = None
        if torch.cuda.is_available():
            device_map = 'cuda'
        elif torch.backends.mps.is_available():
            device_map = 'mps'

        if device_map is not None:
            model = from_pretrained(args.model_name, device_map=device_map, torch_dtype=torch.bfloat16,
                                    batch_size=args.batch_size, height=img_height, width=img_width).eval()
        else:
            model = from_pretrained(args.model_name, batch_size=args.batch_size, height=img_height,
                                    width=img_width).eval()

        generate_tokens(model, args.model_name, args.data_path, inputs_json)
        del model

    if not os.path.exists(model_rkllm):
        from llm import get_model_path
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

        print('Exporting rkllm model ...')
        ret = llm.export_rkllm(model_rkllm)
        if ret != 0:
            print('Export model failed!')
            exit(ret)

    print('Done')


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_name', type=str, default='google/gemma-3n-e2b', help='model name', required=False)
    argparse.add_argument('--opset', type=int, default=19, help='onnx opset', required=False)
    argparse.add_argument('--data_path', type=str, default='data', help='data folder', required=False)
    argparse.add_argument('--out_path', type=str, default='out', help='output folder', required=False)
    argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
    argparse.add_argument('--image_size', type=str, default='480x480',
                          help='image size in format [height x width] or [height,width], default is 480x480',
                          required=False)
    argparse.add_argument('--target_platform', type=str, default='rk3576',
                          help='target platform, choose from [rk3588, rk3576, rk3562, rk3566, rk3568, rk2118, rv1106, rv1103, rv1126b]',
                          required=False)
    argparse.add_argument('--show_onnx', type=bool, default=False, help='show onnx model nodes', required=False)
    argparse.add_argument('--verify_onnx', type=bool, default=False, help='verify onnx model', required=False)
    args = argparse.parse_args()

    main(args)
