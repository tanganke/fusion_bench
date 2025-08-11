fusion_bench \
    method=moe_pruner/moe_pruner \
    method.prune_type=unstructured \
    method.sparsity_ratio=0.5 \
    modelpool=CausalLMPool/mixtral-8x7b \
    merged_model_save_path=outputs/moe_pruner/mixtral-8x7b/unstructured_0.5

fusion_bench \
    method=moe_pruner/moe_pruner \
    method.prune_type=unstructured \
    method.sparsity_ratio=0.7 \
    modelpool=CausalLMPool/mixtral-8x7b \
    merged_model_save_path=outputs/moe_pruner/mixtral-8x7b/unstructured_0.7

fusion_bench \
    method=moe_pruner/moe_pruner \
    method.prune_type=semistructured \
    method.n=2 \
    method.m=4 \
    modelpool=CausalLMPool/mixtral-8x7b \
    merged_model_save_path=outputs/moe_pruner/mixtral-8x7b/semistructured_2_4

fusion_bench \
    method=moe_pruner/moe_pruner \
    method.prune_type=semistructured \
    method.n=4 \
    method.m=8 \
    modelpool=CausalLMPool/mixtral-8x7b \
    merged_model_save_path=outputs/moe_pruner/mixtral-8x7b/semistructured_4_8

# For some GPUs, the following environment variables need to be set
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

function model_eval() {
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

for output_dir in \
    outputs/moe_pruner/mixtral-8x7b/unstructured_0.5 \
    outputs/moe_pruner/mixtral-8x7b/unstructured_0.7 \
    outputs/moe_pruner/mixtral-8x7b/semistructured_2_4 \
    outputs/moe_pruner/mixtral-8x7b/semistructured_4_8; do
    task=gsm8k model_eval
done
