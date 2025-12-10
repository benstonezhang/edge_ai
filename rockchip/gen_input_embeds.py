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
    model = from_pretrained(model_name, load_processor=True, device_map=device_map, dtype=torch_dtype).eval()
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


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='model name', required=True)
    parser.add_argument('--data_dir', type=str, default='data', help='data folder', required=False)
    parser.add_argument('--out_dir', type=str, default='out', help='output folder', required=False)
    parser.add_argument('--force', action='store_true', help='overwrite if output exist')
    return parser.parse_args()


def main():
    args = arg_parser()

    model_name = args.model_name.replace('/', '-').lower()
    data_dir = os.path.realpath(args.data_dir)
    inputs_json = f'{model_name}_inputs.json'

    if args.force or not os.path.exists(os.path.join(args.out_dir, inputs_json)):
        os.makedirs(args.out_dir, mode=0o755, exist_ok=True)
        os.chdir(args.out_dir)
        generate_tokens(model_name=args.model_name, data_dir=data_dir, inputs_json=inputs_json)


if __name__ == '__main__':
    main()
