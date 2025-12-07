import argparse
import os

import torch

from llm.utils import from_pretrained, get_device_and_dtype

argparse = argparse.ArgumentParser()
argparse.add_argument('--model_name', type=str, default=None, help='model name', required=True)
argparse.add_argument('--data_dir', type=str, default='data', help='data folder', required=False)
argparse.add_argument('--batch_size', type=int, default=1, help='batch size', required=False)
argparse.add_argument('--image_path', type=str, default=None,
                      help='image size can be size, or height x width, default is retrieve from model', required=False)
argparse.add_argument('--prompt', type=str, default='Describe this image.', help='prompt text', required=False)
argparse.add_argument('--cpu', action='store_true', help='force run inference on CPU')
args = argparse.parse_args()

image_path = args.image_path if args.image_path is not None else os.path.join(args.data_dir, 'demo.jpg')

if args.cpu:
    device_map, torch_dtype = 'cpu', torch.float32
else:
    device_map, torch_dtype = get_device_and_dtype()

model = from_pretrained(args.model_name, device_map=device_map, dtype=torch_dtype).eval()

if hasattr(model, 'image_token'):
    text_prompt = f'{model.image_token} ${args.prompt}'
else:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text_prompt = model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

image = model.load_image(image_path)
inputs = model.processor(text=text_prompt,
                         images=[image],
                         padding=True,
                         return_tensors='pt').to(model.generation_model.device)
generate_ids = model.generate(**inputs, **model.generation_config)
response = model.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

print(f'User: {args.prompt}\nAssistant: {response}')
