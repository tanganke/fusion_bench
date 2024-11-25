# Introduction to Algorithm Module

The `Fusion Algorithm` module is a core component of the FusionBench project, dedicated to the implementation and execution of various model fusion techniques. 
This module provides the mechanisms necessary to combine multiple models from the Model Pool, enabling nuanced and optimized model merging operations.

### Key Points of the `Fusion Algorithm` Module

- Adaptive Fusion: The module supports advanced fusion techniques, such as AdaMerging, that adaptively learn the best coefficients for model merging using sophisticated methods like entropy minimization.
- Algorithm Configuration: Algorithms are defined and loaded based on configuration files, ensuring flexibility and ease of experimentation. This modular approach allows researchers to switch between different fusion methods seamlessly.
- Model Integration: It facilitates the integration of multiple models, combining their strengths and mitigating individual weaknesses. The result is a single, merged model that ideally performs better than any individual model alone or has multitasking capability.
- Evaluation Support: Once the model fusion process is completed, the merged model can interface with the TaskPool to evaluate the performance of the merged model across various tasks, providing a comprehensive assessment of its capabilities.

#### Example Capabilities

- Entropy Minimization: Some algorithms in this module utilize entropy minimization on unlabeled test samples to refine merging coefficients, ensuring that the fusion process is data-driven and optimized.
- Layer-wise and Task-wise Fusion: It allows both layer-wise and task-wise model fusion, where merging coefficients can be learned for individual layers or entire tasks, respectively.

### Code Integration

The module is typically invoked through a configuration-driven approach in CLI scripts, enabling users to specify fusion algorithms and parameters via YAML configuration files. This method ensures reproducibility and ease of use.
For more information, see [the document of fusion_bench CLI](../cli/fusion_bench.md).

`ModelFusionAlgorithm` is the base class for all fusion algorithms in the Fusion Algorithm module. 
It provides a common interface for different fusion techniques, allowing for seamless integration and execution of various algorithms.


#### Example Usage

Implement your own model fusion algorithm:

```python
from fusion_bench.method import BaseModelFusionAlgorithm
from fusion_bench.modelpool import BaseModelPool

class DerivedModelFusionAlgorithm(BaseModelFusionAlgorithm):
    """
    An example of a derived model fusion algorithm.
    """

    # _config_mapping maps the attribution to the corresponding key in the configuration file.
    _config_mapping = BaseModelFusionAlgorithm._config_mapping | {
        "hyperparam_attr_1": "hyperparam_1",
        "hyperparam_attr_2": "hyperparam_2",
    }

    def __init__(self, hyperparam_1, hyperparam_2, **kwargs):
        self.hyperparam_attr_1 = hyperparam_1
        self.hyperparam_attr_2 = hyperparam_2
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        # implement the fusion algorithm here
        raise NotImplementedError(
            "DerivedModelFusionAlgorithm.run() is not implemented."
        )
```

We provide a simple example to illustrate how the algorithm is used in the FusionBench as follows:

```python
import logging
from typing import Dict, Optional
from omegaconf import DictConfig

from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)

def run_model_fusion(
    method_config: DictConfig,
    modelpool_config: DictConfig,
    taskpool_config: Optional[DictConfig] = None,
    seed: Optional[int] = None,
    print_config: bool = True,
    **kwargs
):
    """
    Run the model fusion process.

    Args:
        method_config: Configuration for the fusion method.
        modelpool_config: Configuration for the model pool.
        taskpool_config: Configuration for the task pool (optional).
    """
    # Instantiate components: modelpool, method, and taskpool
    modelpool = instantiate(modelpool_config)
    method = instantiate(method_config)
    taskpool = None
    if taskpool_config is not None:
        taskpool = instantiate(taskpool_config)

    # Run fusion
    merged_model = method.run(modelpool)

    # Evaluate if taskpool is provided
    if taskpool is not None:
        report = taskpool.evaluate(merged_model)
```

In summary, the Fusion Algorithm module is vital for the model merging operations within FusionBench, leveraging sophisticated techniques to ensure optimal fusion and performance evaluation of deep learning models. This capability makes it an indispensable tool for researchers and practitioners focusing on model fusion strategies.


### References

::: fusion_bench.method.BaseAlgorithm
::: fusion_bench.method.BaseModelFusionAlgorithm
