MODEL_PATH=
OUTPUT_PATH=null

fusion_bench \
    --config-name llama_magnitude_pruning \
    method.prune_type=unstructured \
    method.sparsity_ratio=0.7 \
    modelpool.models.0.path=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}

fusion_bench \
    --config-name llama_magnitude_pruning \
    method.prune_type=semistructured \
    method.n=2 method.m=4 \
    modelpool.models.0.path=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}
