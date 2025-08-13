# Import Algorithm Class from FusionBench

To use FusionBench as a package, you need to import the necessary modules and classes in your Python script.

```python
from torch import nn
from fusion_bench.modelpool import BaseModelPool

def create_mlp(in_features: int, hidden_units: int, out_features: int):
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, out_features)
    )

models = {
    "model_1": create_mlp(768, 3072, 768),
    "model_2": create_mlp(768, 3072, 768),
    "model_3": create_mlp(768, 3072, 768)
}
model_pool = BaseModelPool(models)
```


```python
from fusion_bench.method import SimpleAverageAlgorithm

algorithm = SimpleAverageAlgorithm()
merged_model = algorithm.run(model_pool)
```

