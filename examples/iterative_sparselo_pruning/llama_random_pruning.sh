MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf
OUTPUT_PATH=outputs/llama/random_pruning

function unstructured_pruning() {
    sparsity_ratio=$1
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=pruning/llama_random_pruning \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH} \
        merged_model_save_path=${OUTPUT_PATH}/unstructured/${sparsity_ratio}
}

function semistructured_pruning() {
    n=$1
    m=$2
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=pruning/llama_random_pruning \
        method.prune_type=semistructured \
        method.n=${n} method.m=${m} \
        modelpool.base_model=${MODEL_PATH} \
        merged_model_save_path=${OUTPUT_PATH}/semistructured/${n}_${m}
}

for sparsity_ratio in 0.5 0.7 0.8; do
    unstructured_pruning ${sparsity_ratio}
done

semistructured_pruning 2 4
semistructured_pruning 4 8