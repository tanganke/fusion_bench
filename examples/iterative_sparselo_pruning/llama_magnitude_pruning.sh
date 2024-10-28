export CUDA_VISIBLE_DEVICES=2,3
MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf
OUTPUT_PATH=outputs/llama/magnitude

function unstructured_pruning() {
    _OUTPUT_PATH=${OUTPUT_PATH}/unstructured/${sparsity_ratio}
    if [ -d ${_OUTPUT_PATH} ]; then
        echo "Skip ${_OUTPUT_PATH}"
        return
    fi
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH} \
        merged_model_save_path=${_OUTPUT_PATH}
}

for sparsity_ratio in 0.1 0.2 0.3 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.8; do
    unstructured_pruning
done

fusion_bench  \
    --config-name llama_magnitude_pruning \
    method.prune_type=semistructured \
    method.n=2 method.m=4 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/2_4

fusion_bench \
    --config-name llama_magnitude_pruning \
    method.prune_type=semistructured \
    method.n=4 method.m=8 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/4_8

## llama-13b

export CUDA_VISIBLE_DEVICES="1,2,3"
MODEL_PATH=/data0/users/tanganke/data/huggingface_models/NousResearch/Llama-2-13b-hf
OUTPUT_PATH=outputs/llama-13b/magnitude

function unstructured_pruning() {
    _OUTPUT_PATH=${OUTPUT_PATH}/unstructured/${sparsity_ratio}
    if [ -d ${_OUTPUT_PATH} ]; then
        echo "Skip ${_OUTPUT_PATH}"
        return
    fi
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH} \
        merged_model_save_path=${_OUTPUT_PATH}
}

for sparsity_ratio in 0.5 0.6 0.7; do
    unstructured_pruning
done

fusion_bench \
    --config-name llama_magnitude_pruning \
    method.prune_type=semistructured \
    method.n=2 method.m=4 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/2_4

fusion_bench \
    --config-name llama_magnitude_pruning \
    method.prune_type=semistructured \
    method.n=4 method.m=8 \
    modelpool.base_model=${MODEL_PATH} \
    merged_model_save_path=${OUTPUT_PATH}/semistructured/4_8
