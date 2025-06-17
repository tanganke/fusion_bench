#!/bin/bash

MODEL=$1
GPU_ID=$2
OUTPUT_PATH=$3

echo $MODEL
echo $GPU_ID
echo $OUTPUT_PATH

export CUDA_VISIBLE_DEVICES=$GPU_ID

source $(conda info --base)/etc/profile.d/conda.sh
mkdir -p $OUTPUT_PATH

conda activate lmeval

# category: math
lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks gsm8k_cot \
    --device cuda:$GPU_ID \
    --batch_size 16 \
    --output_path $OUTPUT_PATH

# category: multilingual
lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru \
    --device cuda:$GPU_ID \
    --batch_size 8 \
    --output_path $OUTPUT_PATH

# category: instruction following
lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks ifeval \
    --device cuda:$GPU_ID \
    --batch_size 8 \
    --output_path $OUTPUT_PATH

# category: coding
conda deactivate
conda activate bigcode
cd bigcode-evaluation-harness

accelerate launch main.py \
    --model $MODEL \
    --max_length_generation 512 \
    --precision bf16 \
    --tasks humanevalplus,mbppplus \
    --temperature 0.2 \
    --n_samples 10 \
    --batch_size 10 \
    --allow_code_execution \
    --metric_output_path $OUTPUT_PATH/code_eval.json \
    --use_auth_token

# category: safety
cd ..
conda deactivate
conda activate safety-eval
cd safety-eval-fork

export OPENAI_API_KEY=''

python evaluation/eval.py generators \
    --model_name_or_path $MODEL \
    --use_vllm \
    --model_input_template_path_or_name llama3 \
    --tasks wildguardtest,harmbench,xstest,do_anything_now \
    --report_output_path $OUTPUT_PATH/safety_eval.json \
    --save_individual_results_path $OUTPUT_PATH/safety_generation.json \
    --batch_size 8
