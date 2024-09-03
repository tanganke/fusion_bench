MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf
OUTPUT_PATH=outputs/llama/losparse

for sparsity_ratio in 0.5 0.7 0.8; do
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=llama_losparse \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH} \
        merged_model_save_path=${OUTPUT_PATH}/unstructured/${sparsity_ratio}
done

fusion_bench \
    --config-name llama_magnitude_pruning \
    method=llama_losparse \
        method.prune_type=semistructured \
        method.n=2 method.m=4 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/2_4
