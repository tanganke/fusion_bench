# Model Compression

## Task Vector Compression

### BitDelta

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - BitDeltaAlgorithm

## Parameter Pruning

### Random Pruning

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - RandomPruningForLlama

### Magnitude-based Pruning

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - MagnitudeDiffPruningAlgorithm
        - MagnitudePruningForLlama

### Wanda

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - WandaPruningForLlama

### SparseGPT

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - SparseGPTPruningForLlama

### Pruning with Low-Rank Refinement

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - SparseLoForLlama
        - PCPSparseLoForLlama
        - IterativeSparseLoForLlama

## MoE Expert Pruning

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - DynamicSkippingPruningForMixtral
        - ProgressivePruningForMixtral
        - LayerWisePruningForMixtral
