---
title: Overview
---
# API Reference

Welcome to the FusionBench API reference. This documentation covers all public classes, functions, and modules available in the FusionBench package.

## Quick Links

<div class="grid cards" markdown>

- [:octicons-chevron-right-12: **Algorithms**](fusion_bench.method/index.md) - Fusion algorithms (ensemble, merging, mixing)
- [:octicons-chevron-right-12: **Model Pool**](fusion_bench.modelpool.md) - Model management and loading
- [:octicons-chevron-right-12: **Task Pool**](fusion_bench.taskpool.md) - Evaluation tasks and benchmarks
- [:octicons-chevron-right-12: **Mixins**](fusion_bench.mixins.md) - Reusable functionality
- [:octicons-chevron-right-12: **Programs**](fusion_bench.program.md) - Execution programs
- [:octicons-chevron-right-12: **Utilities**](fusion_bench.utils/index.md) - Helper functions

</div>

---

## Module Overview

### Core Components

| Module | Description |
|--------|-------------|
| [`fusion_bench.method`](fusion_bench.method/index.md) | Fusion algorithms including ensemble, merging, mixing, and compression methods |
| [`fusion_bench.modelpool`](fusion_bench.modelpool.md) | Model pool implementations for managing and loading models |
| [`fusion_bench.taskpool`](fusion_bench.taskpool.md) | Task pool implementations for evaluation |
| [`fusion_bench.mixins`](fusion_bench.mixins.md) | Reusable mixins for extending functionality |
| [`fusion_bench.programs`](fusion_bench.program.md) | Program execution frameworks |

### Supporting Modules

| Module | Description |
|--------|-------------|
| [`fusion_bench.utils`](fusion_bench.utils/index.md) | Utility functions for logging, caching, data processing |
| [`fusion_bench.models`](fusion_bench.models.md) | Model wrapper definitions |
| [`fusion_bench.dataset`](fusion_bench.dataset.md) | Dataset definitions and loaders |
| [`fusion_bench.metrics`](fusion_bench.metrics.md) | Evaluation metrics |
| [`fusion_bench.tasks`](fusion_bench.tasks.md) | Task base classes |
| [`fusion_bench.constants`](fusion_bench.constants.md) | Constant definitions |
| [`fusion_bench.optim`](fusion_bench.optim.md) | Optimizers and learning rate schedulers |

### Deprecated

| Module | Status |
|--------|--------|
| [`fusion_bench.compat`](fusion_bench.compat.md) | Legacy v0.1.x compatibility layer (deprecated) |

---

## Base Classes

### Algorithm Base

The base class for all fusion algorithms in FusionBench.

- [`fusion_bench.BaseAlgorithm`](fusion_bench.method/index.md) - Abstract base class defining the algorithm interface
- [`fusion_bench.BaseModelFusionAlgorithm`](fusion_bench.method/index.md) - Base class for model fusion algorithms

```python
from fusion_bench import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def run(self, modelpool):
        # implement the fusion logic
        pass
```

### Pool Base Classes

- [`fusion_bench.BaseModelPool`](fusion_bench.modelpool.md) - Base class for managing and loading models
- [`fusion_bench.BaseTaskPool`](fusion_bench.taskpool.md) - Base class for evaluation tasks

### Program Base

- [`fusion_bench.BaseHydraProgram`](fusion_bench.program.md) - Base class for Hydra-based execution programs
- [`fusion_bench.FabricModelFusionProgram`](fusion_bench.program.md) - Program for fusing models with Lightning Fabric

---

## Usage Examples

### Using an Algorithm

```python
from fusion_bench import SimpleAverageAlgorithm

# Create algorithm instance
algorithm = SimpleAverageAlgorithm(
    weight_key="task_weight",
    scale_factor=1.0
)

# Run fusion
merged_model = algorithm.run(modelpool)
```

### Using Model Pool

```python
from fusion_bench import CLIPVisionModelPool

# Load models from pool
modelpool = CLIPVisionModelPool.from_config("config/modelpool/CLIPVisionModelPool/clip-vit-base-patch32_TA8.yaml")
models = modelpool.load_models()
```

### Using Task Pool

```python
from fusion_bench import CLIPVisionModelTaskPool

# Evaluate model on tasks
taskpool = CLIPVisionModelTaskPool()
results = taskpool.evaluate(model)
```

---

## Navigation

Browse the API documentation by category:

- [Model Ensemble Algorithms](fusion_bench.method/ensemble.md) - Simple and weighted ensembles
- [Model Merging Algorithms](fusion_bench.method/merging.md) - Linear interpolation and optimization-based methods
- [Model Mixing Algorithms](fusion_bench.method/mixing.md) - Layer-level and MoE-based mixing
- [Model Compression Algorithms](fusion_bench.method/compression.md) - Pruning and compression techniques
- [Training Algorithms](fusion_bench.method/training.md) - Fine-tuning and training methods
