#!/usr/bin/env bash

#set -e

cwd=$(realpath "$(dirname "$0")/..")
cd "$cwd" || exit

declare -A LEGACY_MODELS
declare -A MODELS

LEGACY_MODELS[qwen2]='rknn'
LEGACY_MODELS[smolvlm2]='rkllm'

if [ $# -lt 2 ]; then
    MODELS[qwen2]='Qwen/Qwen2-VL-2B-Instruct'
    MODELS[qwen2.5]='Qwen/Qwen2.5-VL-3B-Instruct'
    MODELS[qwen3]='Qwen/Qwen3-VL-4B-Instruct'
    MODELS[internvl3]='OpenGVLab/InternVL3-2B-Instruct'
    MODELS[smolvlm2]='HuggingFaceTB/SmolVLM2-2.2B-Instruct'
    MODELS[minicpm_v_2.6]='OpenBMB/MiniCPM-V-2_6'
    MODELS[gemma3]='google/gemma-3-4b-it'
    #MODELS[gemma3n]='google/gemma-3n-E2B-it'
else
    while [ $# -ge 2 ]; do
        echo "$1 -> $2"
        MODELS[$1]="$2"
        shift 2
    done
fi

if [ -d "$cwd/.pyenv" ]; then
    PYENV_ROOT="$cwd/.pyenv"
elif [ -d "$HOME/.pyenv/versions" ]; then
    PYENV_ROOT="$HOME/.pyenv/versions"
else
    PYENV_ROOT="$HOME/.pyenv"
fi
export PYTHONPATH="$cwd"

source "${PYENV_ROOT}/llm/bin/activate"
for m in "${!MODELS[@]}"; do
    [ -n "${LEGACY_MODELS[$m]}" ] && continue
    python rockchip/export_vision.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
    python rockchip/gen_input_embeds.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
done
deactivate

source "${PYENV_ROOT}/rknn/bin/activate"
for m in "${!MODELS[@]}"; do
    if  [ "_${LEGACY_MODELS[$m]}" = '_rknn' ]; then
        python rockchip/export_vision.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
    fi
    python rockchip/export_rknn.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
done
deactivate

source "${PYENV_ROOT}/rkllm/bin/activate"
for m in "${!MODELS[@]}"; do
    if  [ "_${LEGACY_MODELS[$m]}" = '_rkllm' ]; then
        python rockchip/export_vision.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
    fi
    [ -n "${LEGACY_MODELS[$m]}" ] && python rockchip/gen_input_embeds.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
    python rockchip/export_rkllm.py --model_name="${MODELS[$m]}" --out_dir="out/$m"
done
deactivate
