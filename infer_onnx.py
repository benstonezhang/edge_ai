import argparse
import os

import numpy as np
import onnxruntime
from transformers import AutoConfig, AutoProcessor

from llm.utils import get_device_and_dtype, get_model_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='google/gemma-3n-E2B-it', help='model name', required=False)
parser.add_argument('--model_dir', type=str, default='onnx-community/gemma-3n-E2B-it-ONNX',
                    help='onnx model folder', required=False)
parser.add_argument('--embed_tokens', type=str, default='embed_tokens_quantized.onnx',
                    help='embed_tokens model file name', required=False)
parser.add_argument('--audio_encoder', type=str, default='audio_encoder_quantized.onnx',
                    help='audio_encoder model file name', required=False)
parser.add_argument('--audio_path', type=str, default=None, help='audio file path', required=False)
parser.add_argument('--vision_encoder', type=str, default='vision_encoder_quantized.onnx',
                    help='vision_encoder model file name', required=False)
parser.add_argument('--image_path', type=str, default=None, help='image file path', required=False)
parser.add_argument('--decoder_model_merged', type=str, default='decoder_model_merged_q4.onnx',
                    help='decoder_model_merged model file name', required=False)
args = parser.parse_args()

model_dir = args.model_dir
if not os.path.exists(model_dir):
    model_dir = os.path.join(get_model_path(model_dir), 'onnx')

audio_path = os.path.expanduser("~/Downloads/huggingface_jfk.wav") if args.audio_path is None else args.audio_path
image_path = os.path.expanduser("~/Downloads/huggingface_bee.jpg") if args.image_path is None else args.image_path

device_map, torch_dtype = get_device_and_dtype()

# 1. Load models
## Load config and processor
processor = AutoProcessor.from_pretrained(args.model_name, device_map=device_map, dtype=torch_dtype)
config = AutoConfig.from_pretrained(args.model_name, device_map=device_map, dtype=torch_dtype)

## Load sessions
embed_model_path = os.path.join(model_dir, args.embed_tokens)
audio_model_path = os.path.join(model_dir, args.audio_encoder)
vision_model_path = os.path.join(model_dir, args.vision_encoder)
decoder_model_path = os.path.join(model_dir, args.decoder_model_merged)

vision_session = onnxruntime.InferenceSession(vision_model_path)
audio_session = onnxruntime.InferenceSession(audio_model_path)
embed_session = onnxruntime.InferenceSession(embed_model_path)
decoder_session = onnxruntime.InferenceSession(decoder_model_path)

## Set config values
num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
num_hidden_layers = config.text_config.num_hidden_layers
eos_token_id = 106  # != config.text_config.eos_token_id
image_token_id = config.image_token_id
audio_token_id = config.audio_token_id

# 2. Prepare inputs
## Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "In detail, describe the following audio and image."},
            {"type": "audio", "audio": audio_path},
            {"type": "image", "image": image_path},
        ],
    },
]
# text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# <bos><start_of_turn>user\nIn detail, describe the following audio and image.<audio_soft_token><image_soft_token><end_of_turn>\n<start_of_turn>model
inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
)
input_ids = inputs["input_ids"].numpy()
attention_mask = inputs["attention_mask"].numpy()
position_ids = np.cumsum(attention_mask, axis=-1) - 1

pixel_values = inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
input_features = inputs["input_features"].numpy().astype(np.float32) if "input_features" in inputs else None
input_features_mask = inputs["input_features_mask"].numpy() if "input_features_mask" in inputs else None

## Prepare decoder inputs
batch_size = input_ids.shape[0]
past_key_values = {
    f"past_key_values.{layer}.{kv}": np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ("key", "value")
}

# 3. Generation loop
max_new_tokens = 1024
generated_tokens = np.array([[]], dtype=np.int64)
image_features = None
audio_features = None
for i in range(max_new_tokens):
    inputs_embeds, per_layer_inputs = embed_session.run(None, {"input_ids": input_ids})
    if image_features is None and pixel_values is not None:
        image_features = vision_session.run(
                ["image_features"],
                {
                    "pixel_values": pixel_values,
                }
        )[0]
        mask = (input_ids == image_token_id).reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds[mask] = image_features.reshape(-1, image_features.shape[-1])
        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

    if audio_features is None and input_features is not None and input_features_mask is not None:
        audio_features = audio_session.run(
                ["audio_features"],
                {
                    "input_features": input_features,
                    "input_features_mask": input_features_mask,
                }
        )[0]
        mask = (input_ids == audio_token_id).reshape(-1)
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat_embeds[mask] = audio_features.reshape(-1, audio_features.shape[-1])
        inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

    logits, *present_key_values = decoder_session.run(None, dict(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            **past_key_values,
    ))

    ## Update values for next generation loop
    input_ids = logits[:, -1].argmax(-1, keepdims=True)
    attention_mask = np.ones_like(input_ids)
    position_ids = position_ids[:, -1:] + 1
    for j, key in enumerate(past_key_values):
        past_key_values[key] = present_key_values[j]

    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
    if (input_ids == eos_token_id).all():
        break

    ## (Optional) Streaming
    print(processor.decode(input_ids[0]), end="", flush=True)
print()

# 4. Output result
print(processor.batch_decode(generated_tokens, skip_special_tokens=True)[0])
