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
