# Introduction

The `Fusion Algorithm` module is a core component of the FusionBench project, dedicated to the implementation and execution of various model fusion techniques. 
This module provides the mechanisms necessary to combine multiple models from the Model Pool, enabling nuanced and optimized model merging operations.

### Key Points of the `Fusion Algorithm` Module

- Adaptive Fusion: The module supports advanced fusion techniques, such as AdaMerging, that adaptively learn the best coefficients for model merging using sophisticated methods like entropy minimization.
- Algorithm Configuration: Algorithms are defined and loaded based on configuration files, ensuring flexibility and ease of experimentation. This modular approach allows researchers to switch between different fusion methods seamlessly.
- Model Integration: It facilitates the integration of multiple models, combining their strengths and mitigating individual weaknesses. The result is a single, merged model that ideally performs better than any individual model alone or have multitasking capability.
- Evaluation Support: Once the model fusion process is completed, the merged model can interface with the TaskPool to evaluate the performance of the merged model across various tasks, providing a comprehensive assessment of its capabilities.

#### Example Capabilities

- Entropy Minimization: Some algorithms in this module utilize entropy minimization on unlabeled test samples to refine merging coefficients, ensuring that the fusion process is data-driven and optimized.
- Layer-wise and Task-wise Fusion: It allows both layer-wise and task-wise model fusion, where merging coefficients can be learned for individual layers or entire tasks, respectively.

### Code Integration

The module is typically invoked through a configuration-driven approach in CLI scripts, enabling users to specify fusion algorithms and parameters via YAML configuration files. This method ensures reproducibility and ease of use.
For more information, see [the document of fusion_bench CLI](/cli/fusion_bench).

`ModelFusionAlgorithm` is the base class for all fusion algorithms in the Fusion Algorithm module. 
It provides a common interface for different fusion techniques, allowing for seamless integration and execution of various algorithms.

::: fusion_bench.method.base_algorithm.ModelFusionAlgorithm

#### Example Usage

```python
from ..method import load_algorithm_from_config
from ..modelpool import load_modelpool_from_config

def run_model_fusion(cfg: DictConfig):
    modelpool = load_modelpool_from_config(cfg.modelpool)
    algorithm = load_algorithm_from_config(cfg.method)
    merged_model = algorithm.fuse(modelpool)

    if hasattr(cfg, "taskpool") and cfg.taskpool is not None:
        taskpool = load_taskpool_from_config(cfg.taskpool)
        taskpool.evaluate(merged_model)
    else:
        print("No task pool specified. Skipping evaluation.")
```

In summary, the Fusion Algorithm module is vital for the model merging operations within FusionBench, leveraging sophisticated techniques to ensure optimal fusion and performance evaluation of deep learning models. This capability makes it an indispensable tool for researchers and practitioners focusing on model fusion strategies.
