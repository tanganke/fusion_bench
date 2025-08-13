# Import and Use Merging Algorithms

This tutorial demonstrates how to import and use different model merging algorithms from FusionBench as a Python package. You'll learn how to programmatically create model pools, apply various fusion algorithms, and obtain merged models without using the CLI interface.

## 🚀 Quick Start

### Creating a Model Pool

First, let's create a simple model pool with multiple models that we want to merge:

```python linenums="1" hl_lines="2 20"
from torch import nn
from fusion_bench.modelpool import BaseModelPool

def create_mlp(in_features: int, hidden_units: int, out_features: int):
    """Create a simple multi-layer perceptron."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, out_features)
    )

# Create multiple models with the same architecture
models = {
    "model_1": create_mlp(768, 3072, 768),
    "model_2": create_mlp(768, 3072, 768),
    "model_3": create_mlp(768, 3072, 768)
}

# Create a model pool
model_pool = BaseModelPool(models)
```

The simplest approach is to use the Simple Average algorithm, which averages the parameters of all models:

```python linenums="1"
from fusion_bench.method import SimpleAverageAlgorithm

# Initialize the algorithm
algorithm = SimpleAverageAlgorithm()

# Merge the models
merged_model = algorithm.run(model_pool)

print(f"Successfully merged {len(models)} models!")
```

## 💡 More Examples

FusionBench provides various merging algorithms. Here are some commonly used ones:

### 1. Simple Average

Averages all model parameters equally - no hyperparameters needed:

```python
from fusion_bench.method import SimpleAverageAlgorithm

algorithm = SimpleAverageAlgorithm()
merged_model = algorithm.run(model_pool)
```

### 2. Weighted Average

Allows you to assign different weights to each model:

```python
from fusion_bench.method import WeightedAverageAlgorithm

# Define weights for each model (must sum to 1.0)
weights = [0.5, 0.3, 0.2]

algorithm = WeightedAverageAlgorithm(
    weights=weights,
    normalize=True  # Automatically normalize weights to sum to 1
)
merged_model = algorithm.run(model_pool)
```

### 3. Task Arithmetic

Enables task arithmetic operations with a scaling factor:

```python
from fusion_bench.method import TaskArithmeticAlgorithm

algorithm = TaskArithmeticAlgorithm(
    scaling_factor=0.3,  # Controls the strength of task vectors
)

# Create multiple models with the same architecture
models = {
    # To compute the task vectors, we need a pretrained model
    "_pretrained_": create_mlp(768, 3072, 768),
    "model_1": create_mlp(768, 3072, 768),
    "model_2": create_mlp(768, 3072, 768),
    "model_3": create_mlp(768, 3072, 768)
}
model_pool = BaseModelPool(models)

merged_model = algorithm.run(model_pool)
```
