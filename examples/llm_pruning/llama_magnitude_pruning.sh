MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf
OUTPUT_PATH=null

fusion_bench \
    --config-name llama_magnitude_pruning \
    method.prune_type=unstructured \
    method.sparsity_ratio=0.7 \
    modelpool.models._pretrained_.pretrained_model_name_or_path=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}

fusion_bench \
    --config-name llama_magnitude_pruning \
    method.prune_type=semistructured \
    method.n=2 method.m=4 \
    modelpool.models._pretrained_.pretrained_model_name_or_path=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}
