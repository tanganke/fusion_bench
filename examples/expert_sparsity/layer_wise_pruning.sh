for calib_set in c4 math; do
    for r in 4 6; do
        fusion_bench \
            fabric.loggers.name="mixtral_8x7b_expert_pruning/layer_wise_pruning" \
            fabric.loggers.version="${calib_set}" \
            fabric.loggers.sub_dir="${r}_experts" \
            method=expert_sparsity/mixtral \
            method._target_=fusion_bench.method.LayerWisePruningForMixtral \
            method.num_preserved_experts=${r} \
            modelpool=CausalLMPool/mixtral-8x7b
    done
done
