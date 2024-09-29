#! /usr/bin/env bash
# For some GPUs, the following environment variables need to be set
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

function model_eval() {
    # Usage:
    # model_eval <model_dir> <task>
    output_dir=$1
    task=$2

    # Check if ${output_dir}/${task}.json exists as a directory and return if it does
    if [ -d "${output_dir}/${task}.json" ]; then
        echo "Directory ${output_dir}/${task}.json already exists. Skipping evaluation."
        return
    fi

    lm_eval --model hf \
        --model_args pretrained=${output_dir},dtype="float16",parallelize=True \
        --tasks ${task} \
        --output_path ${output_dir}/${task}.json \
        --batch_size 6
}

model_dir=/data0/users/tanganke/projects/fusion_bench/outputs/llama/decapoda-research-llama-7b-hf
for task in truthfulqa gsm8k arc_challenge mmlu; do
    model_eval $model_dir $task
done

for model_dir in \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama/decapoda-research-llama-7b-hf \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama/magnitude/unstructured/0.5 \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama/sparselo/magnitude/unstructured/0.5 \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama/iterative_sparselo/magnitude/unstructured/0.5; do
    for task in truthfulqa gsm8k arc_challenge mmlu; do
        model_eval $model_dir $task
    done
done

for model_dir in \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama-13b/Llama-2-13b-hf \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama-13b/magnitude/semistructured/2_4 \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama-13b/sparselo/magnitude/semistructured/2_4 \
    /data0/users/tanganke/projects/fusion_bench/outputs/llama-13b/iterative_sparselo/magnitude/semistructured/2_4 \
    ; do
    for task in truthfulqa gsm8k arc_challenge mmlu; do
        model_eval $model_dir $task
    done
done
