export CUDA_VISIBLE_DEVICES="5"

MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf
OUTPUT_PATH=outputs/llama/wanda_pruning

for sparsity_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    if [ -d ${OUTPUT_PATH}/unstructured/${sparsity_ratio} ]; then
        echo "Skip ${OUTPUT_PATH}/unstructured/${sparsity_ratio}"
        continue
    fi
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=pruning/llama_wanda_pruning \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH} \
        merged_model_save_path=${OUTPUT_PATH}/unstructured/${sparsity_ratio}
done

fusion_bench \
    --config-name llama_magnitude_pruning \
    method=pruning/llama_wanda_pruning \
    method.prune_type=semistructured \
    method.n=2 method.m=4 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/2_4

fusion_bench \
    --config-name llama_magnitude_pruning \
    method=pruning/llama_wanda_pruning \
    method.prune_type=semistructured \
    method.n=4 method.m=8 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/4_8

#  llama 13b
export CUDA_VISIBLE_DEVICES="5,6,7"

MODEL_PATH=/data0/users/tanganke/data/huggingface_models/NousResearch/Llama-2-13b-hf
OUTPUT_PATH=outputs/llama-13b/wanda_pruning

for sparsity_ratio in 0.5; do
    if [ -d ${OUTPUT_PATH}/unstructured/${sparsity_ratio} ]; then
        echo "Skip ${OUTPUT_PATH}/unstructured/${sparsity_ratio}"
        continue
    fi
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=pruning/llama_wanda_pruning \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH} \
        merged_model_save_path=${OUTPUT_PATH}/unstructured/${sparsity_ratio}
done

fusion_bench \
    --config-name llama_magnitude_pruning \
    method=pruning/llama_wanda_pruning \
    method.prune_type=semistructured \
    method.n=2 method.m=4 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/2_4

fusion_bench \
    --config-name llama_magnitude_pruning \
    method=pruning/llama_wanda_pruning \
    method.prune_type=semistructured \
    method.n=4 method.m=8 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/4_8
