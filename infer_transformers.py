import argparse
import os

import torch
from transformers import AutoProcessor

from llm.utils import bare_model, from_pretrained, get_device_and_dtype

argparse = argparse.ArgumentParser()
argparse.add_argument('--model_name', type=str, default=None, help='model name', required=True)
argparse.add_argument('--data_path', type=str, default='data', help='data folder', required=False)
argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
argparse.add_argument('--image_size', type=str, default=None,
                      help='image size in format [height x width], default is retrieve from model', required=False)
argparse.add_argument('--image_path', type=str, default=None, help='image path', required=False)
argparse.add_argument('--prompt', type=str, default='Describe this image.', help='prompt text', required=False)
argparse.add_argument('--cpu', action='store_true', help='force run inference on CPU')
args = argparse.parse_args()

if args.image_size:
    img_height, img_width = (int(x) for x in args.image_size.split('x'))
else:
    model = bare_model(args.model_name)
    if model.image_size is None:
        print('Please specify image height and width')
        exit(1)
    img_height, img_width = model.image_size
    del model

image_path = args.image_path if args.image_path is not None else os.path.join(args.data_path, 'demo.jpg')

if args.cpu:
    device_map, torch_dtype = 'cpu', torch.float32
else:
    device_map, torch_dtype = get_device_and_dtype()
model = from_pretrained(args.model_name, device_map=device_map, torch_dtype=torch_dtype,
                        batch_size=args.batch_size, height=img_height, width=img_width).eval()
processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True, **model.tokenizer_config)

if hasattr(model.generation_model, 'chat'):
    pixel_values = model.load_image_tensor(image_path).to(model.generation_model.device, model.generation_model.dtype)
    response = model.generation_model.chat(processor, pixel_values, args.prompt, model.generation_config)
else:
    if hasattr(processor, 'apply_chat_template'):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    else:
        text_prompt = f'{model.image_token} ${args.prompt}'

    if hasattr(model, 'replace_image_tokens'):
        text_prompt = model.replace_image_tokens(processor, text_prompt)
        inputs = processor(text=text_prompt, padding=True, return_tensors='pt').to(model.generation_model.device)
        inputs['pixel_values'] = model.load_image_tensor(image_path).to(model.generation_model.device,
                                                                        model.generation_model.dtype)
    else:
        inputs = processor(text=text_prompt,
                           images=[model.load_image(image_path)],
                           padding=True,
                           return_tensors='pt').to(model.generation_model.device)

    generate_ids = model.generate(**inputs, **model.generation_config)
    if model.output_prefix_with_prompt:
        input_len = inputs["input_ids"].shape[-1]
        generate_ids = generate_ids[0][input_len:].unsqueeze(0)

    response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

print(f'User: {args.prompt}\nAssistant: {response}')
