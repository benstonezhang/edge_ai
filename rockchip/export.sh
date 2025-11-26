#!/usr/bin/env bash

cwd=$(realpath "$(dirname "$0")/..")
cd "$cwd" || exit

export PYTHONPATH="$cwd"
declare -A MODELS
MODELS[qwen2]='Qwen/Qwen2-VL-2B-Instruct'
MODELS[qwen2.5]='Qwen/Qwen2.5-VL-3B-Instruct'
MODELS[internvl3]='OpenGVLab/InternVL3-2B-Instruct'
MODELS[smolvlm2]='HuggingFaceTB/SmolVLM2-2.2B-Instruct'
MODELS[minicpm_v_2.6]='OpenBMB/MiniCPM-V-2_6'
MODELS[gemma3]='google/gemma-3-4b-it'
#MODELS[gemma3n]='google/gemma-3n-E2B-it'
set -ex

pyenv activate rknn
for m in "${!MODELS[@]}"; do
    python rockchip/export_rknn.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
done
pyenv deactivate

pyenv activate rkllm
for m in "${!MODELS[@]}"; do
    python rockchip/export_rkllm.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
done
pyenv deactivate
