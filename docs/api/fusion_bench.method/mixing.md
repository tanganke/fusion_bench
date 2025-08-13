# Model Mixing

## Layer-level Mixing

### Depth Upscaling

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - DepthUpscalingAlgorithm
        - DepthUpscalingForLlama

### Model Recombination

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - ModelRecombinationAlgorithm

::: fusion_bench.method.model_recombination
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - recombine_modellist
        - recombine_modeldict
        - recombine_state_dict

## MoE-based Mixing

### MoE Upscaling

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - MixtralUpscalingAlgorithm
        - MixtralForCausalLMUpscalingAlgorithm
        - MixtralMoEMergingAlgorithm
        - MixtralForCausalLMMergingAlgorithm

### Weight-Ensembling Mixture of Experts (WE-MoE)

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - CLIPWeightEnsemblingMoEAlgorithm

### Sparse WE-MoE

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - SparseWeightEnsemblingMoEAlgorithm
        - SparseCLIPWeightEnsemblingMoEAlgorithm

### Rank-One MoE

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - RankOneMoEAlgorithm
        - CLIPRankOneMoEAlgorithm

### Pareto-driven Weight-Ensembling MoE (PWE-MoE)

::: fusion_bench.method.pwe_moe.clip_pwe_moe
      options:
        show_root_heading: false
        heading_level: 4
        members:
        - PWEMoEAlgorithmForCLIP
        - PWEMoELinearScalarizationForCLIP
        - PWEMoExactParetoOptimalForCLIP

### Smile Upscaling

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - SmileUpscalingAlgorithm
        - SingularProjectionMergingAlgorithm
