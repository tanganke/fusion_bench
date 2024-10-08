export CUDA_VISIBLE_DEVICES=2,3,4
# MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf
MODEL_PATH=/data0/users/tanganke/data/huggingface_models/NousResearch/Llama-2-13b-hf

function unstructured_pruning() {
    sparsity_ratio=$1
    OUTPUT_PATH=outputs/llama-13b/sparselo/${VARIANT}
    if [ -d ${OUTPUT_PATH}/unstructured/${sparsity_ratio} ]; then
        echo "Skip ${OUTPUT_PATH}/unstructured/${sparsity_ratio}"
        return
    fi
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=sparselo_pruning/llama_sparselo \
        method.variant=${VARIANT} \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        method.model_save_path=${OUTPUT_PATH}/unstructured/${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH}
}

function semistructured_pruning() {
    n=$1
    m=$2
    OUTPUT_PATH=outputs/llama-13b/sparselo/${VARIANT}
    if [ -d ${OUTPUT_PATH}/semistructured/${n}_${m} ]; then
        echo "Skip ${OUTPUT_PATH}/semistructured/${n}_${m}"
        return
    fi
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=sparselo_pruning/llama_sparselo \
        method.variant=${VARIANT} \
        method.prune_type=semistructured \
        method.n=${n} method.m=${m} \
        method.model_save_path=${OUTPUT_PATH}/semistructured/${n}_${m} \
        modelpool.base_model=${MODEL_PATH}
}

# wanda-sparselo
VARIANT=wanda
for sparsity_ratio in 0.5 0.6 0.7; do
    unstructured_pruning ${sparsity_ratio}
done

semistructured_pruning 2 4
semistructured_pruning 4 8

# magnitude-sparselo
VARIANT=magnitude

for sparsity_ratio in 0.5 0.6 0.7; do
    unstructured_pruning ${sparsity_ratio}
done

semistructured_pruning 2 4
semistructured_pruning 4 8
