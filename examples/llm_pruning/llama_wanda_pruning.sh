MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf
OUTPUT_PATH=outputs/llama/wanda_pruning

echo fusion_bench \
    --config-name llama_magnitude_pruning \
    method=llama_wanda_pruning \
    method.prune_type=unstructured \
    method.sparsity_ratio=0.7 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/unstructured/0.7

fusion_bench \
    --config-name llama_magnitude_pruning \
    method=llama_wanda_pruning \
    method.prune_type=semistructured \
    method.n=2 method.m=4 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/2_4
