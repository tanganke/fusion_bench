# Parallel CLIP Ensemble

This tutorial demonstrates how to create and evaluate a parallel ensemble of CLIP (Contrastive Language-Image Pre-training) models using device mapping for efficient multi-GPU inference. Unlike model fusion techniques that merge parameters, ensemble methods maintain separate models and aggregate their predictions at inference time.

The ensemble approach averages predictions from multiple fine-tuned models:

\[
y_{ensemble} = \frac{1}{N} \sum_{i=1}^{N} f_i(x)
\]

where \( y_{ensemble} \) is the ensemble prediction, \( N \) is the number of models, and \( f_i(x) \) is the prediction from the i-th model.

## 🚀 Key Features

- **Parallel Execution**: Models run simultaneously on different GPUs using `torch.jit.fork`
- **Device Mapping**: Distribute models across multiple devices for memory efficiency
- **Automatic Synchronization**: Outputs are automatically moved to the same device for aggregation

## 🔧 Python Implementation

Here's a complete example demonstrating parallel ensemble evaluation:

```python title="examples/ensemble/parallel_ensemble.py" linenums="1"
import os

import lightning as L
from hydra import compose, initialize

from fusion_bench import instantiate
from fusion_bench.method import SimpleEnsembleAlgorithm
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.rich_utils import setup_colorlogging

setup_colorlogging()

fabric = L.Fabric(accelerator="auto", devices=1)

# Load configuration using Hydra
with initialize(
    version_base=None,
    config_path=os.path.relpath(
        _get_default_config_path(), start=os.path.dirname(__file__)
    ),
):
    cfg = compose(
        config_name="fabric_model_fusion",
        overrides=[
            "modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8",
            "taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8",
        ],
    )
    modelpool: CLIPVisionModelPool = instantiate(cfg.modelpool)
    taskpool: CLIPVisionModelTaskPool = instantiate(cfg.taskpool, move_to_device=False)
    taskpool.fabric = fabric

# Hard-coded device map and algorithm instantiation for 8 models
device_map = {
    0: "cuda:0",
    1: "cuda:0",
    2: "cuda:0",
    3: "cuda:0",
    4: "cuda:1",
    5: "cuda:1",
    6: "cuda:1",
    7: "cuda:1",
}
algorithm = SimpleEnsembleAlgorithm(device_map=device_map)
ensemble = algorithm.run(modelpool)

report = taskpool.evaluate(ensemble)
print(report)
```

## 🔧 YAML Configuration

Alternatively, you can use the ensemble method configurations:

```yaml title="config/method/ensemble/simple_ensemble.yaml"
--8<-- "config/method/ensemble/simple_ensemble.yaml"
```

## 🚀 Running the Example

Execute the parallel ensemble evaluation:

```bash
cd examples/ensemble
python parallel_ensemble.py
```
