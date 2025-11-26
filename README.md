# edge_ai

Tools to deploy LLM to edge devices

# Usage

`mtmd [options] image_path encoder_model_path llm_model_path`

| options           | description                                                    |
|-------------------|----------------------------------------------------------------|
| --core_num        | NPU core number: 2 for rk3588, 2 for rt3576, 1 for others      |
| --max_context_len | max of total context length, default is 4095                   |
| --max_new_tokens  | max tokens the model will generate, default is 256             |
| --chat_template   | chat template file if rkllm can't retrieve from model          |
| --img_tokens      | default is <\|vision_start\|>,<\|vision_end\|>,<\|image_pad\|> |
| --img_size        | optional image size as format Height,Weight                    |

Notes:

google/gemma-3 need add "`--img_tokens="<start_of_image>,<image_soft_token>,<end_of_image>`"

HuggingFaceTB/SmolVLM2 need add "`--chat_template`"
