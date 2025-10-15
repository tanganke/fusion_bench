# Model Merging

## Linear Interpolation

### Simple Average

::: fusion_bench.method.simple_average.simple_average
    options:
        heading_level: 4

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - SimpleAverageAlgorithm
        - SimpleAverageForLlama

### Weighted Average

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - LinearInterpolationAlgorithm
        - WeightedAverageAlgorithm
        - WeightedAverageForLLama

### Spherical Linear Interpolation (SLERP)

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - SlerpMergeAlgorithm
        - SlerpForCausalLM

### Task Arithmetic

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - TaskArithmeticAlgorithm
        - TaskArithmeticForLlama

### Ties-Merging

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - TiesMergingAlgorithm

### Fisher Merging

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - FisherMergingAlgorithm
        - FisherMergingForCLIPVisionModel
        - FisherMergingAlgorithmForGPT2

### Drop And REscale (DARE)

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - DareSimpleAverage
        - DareTaskArithmetic
        - DareTiesMerging

### Model Extrapolation (ExPO)

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - ExPOAlgorithm
        - ExPOAlgorithmForLlama

### DOGE

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - DOGE_TA_Algorithm

### AdaMerging

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - CLIPTaskWiseAdaMergingAlgorithm
        - CLIPLayerWiseAdaMergingAlgorithm
        - GPT2LayerWiseAdaMergingAlgorithm
        - FlanT5LayerWiseAdaMergingAlgorithm


## Optimization-based Methods

### RegMean

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - RegMeanAlgorithmForCLIP
        - RegMeanAlgorithmForGPT2

### RegMean++

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - RegMeanAlgorithmPlusPlus
        - RegMeanAlgorithmForCLIPPlusPlus

### Frank-Wolfe Merging

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - FrankWolfeSoftAlgorithm
        - FrankWolfeHardAlgorithm

### WUDI-Merging

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - WUDIMerging

## Subspace-based Methods

### Concrete Subspace

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - ConcreteTaskArithmeticAlgorithmForCLIP
        - ConcreteTaskWiseAdaMergingForCLIP
        - ConcreteLayerWiseAdaMergingForCLIP

### Task Singular Vector Merging (TSVM)

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - TaskSingularVectorMerging

### Isotropic Merging

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - ISO_C_Merge
        - ISO_CTS_Merge
        - IsotropicMergingInCommonSubspace
        - IsotropicMergingInCommonAndTaskSubspace

## Distributed Model Merging

### Gossip

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - CLIPTaskWiseGossipAlgorithm
        - CLIPLayerWiseGossipAlgorithm
        - FlanT5LayerWiseGossipAlgorithm

## Continual Model Merging

### Orthogonal Projection-based Continual Merging (OPCM)

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - OPCMForCLIP

### Dual Projections (DOP)

::: fusion_bench.method
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - ContinualDOPForCLIP
