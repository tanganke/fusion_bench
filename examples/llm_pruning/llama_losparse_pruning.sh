export CUDA_VISIBLE_DEVICES=6,7
MODEL_PATH=/data0/users/tanganke/data/huggingface_models/decapoda-research/llama-7b-hf

function unstructured_pruning() {
    sparsity_ratio=$1
    OUTPUT_PATH=outputs/llama/losparse/${VARIANT}
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=llama_losparse \
        method.variant=${VARIANT} \
        method.prune_type=unstructured \
        method.sparsity_ratio=${sparsity_ratio} \
        method.model_save_path=${OUTPUT_PATH}/unstructured/${sparsity_ratio} \
        modelpool.base_model=${MODEL_PATH}
}

function semistructured_pruning() {
    n=$1
    m=$2
    OUTPUT_PATH=outputs/llama/losparse/${VARIANT}
    fusion_bench \
        --config-name llama_magnitude_pruning \
        method=llama_losparse \
        method.variant=${VARIANT} \
        method.prune_type=semistructured \
        method.n=${n} method.m=${m} \
        method.model_save_path=${OUTPUT_PATH}/semistructured/${n}_${m} \
        modelpool.base_model=${MODEL_PATH}
}

# dense
VARIANT="dense"
unstructured_pruning 0

# lowrank-only
VARIANT="lowrank-only"
unstructured_pruning 1

# wanda-losparse
VARIANT=wanda
for sparsity_ratio in 0.5 0.7 0.8; do
    unstructured_pruning ${sparsity_ratio}
done

semistructured_pruning 2 4
semistructured_pruning 4 8

# magnitude-losparse
VARIANT=magnitude

for sparsity_ratio in 0.5 0.7 0.8; do
    unstructured_pruning ${sparsity_ratio}
done

semistructured_pruning 2 4
semistructured_pruning 4 8

# random-losparse
VARIANT=random

for sparsity_ratio in 0.5 0.7 0.8; do
    unstructured_pruning ${sparsity_ratio}
done

semistructured_pruning 2 4
semistructured_pruning 4 8
