# Weighted Ensemble

A weighted ensemble is a machine learning technique that combines the predictions of multiple models to produce a final prediction. The idea is to leverage the strengths of each individual model to improve overall performance and robustness.

Formally, a weighted ensemble can be defined as follows:

Given a set of $n$ models, each model $f_i$ produces a prediction $f_i(x)$ for an input $x$. Each model $i$ also has an associated weight $w_i$. The final prediction $F(x)$ of the weighted ensemble is a weighted sum of the individual model predictions:

$$
F(x) = w_1 f_1(x) + w_2 f_2(x) + ... + w_n f_n(x)
$$

The weights $w_i$ are typically non-negative and sum to 1 (i.e., $\sum_{i=1}^n w_i = 1$), which ensures that the final prediction is a convex combination of the individual model predictions.
The weights can be determined in various ways. They could be set based on the performance of the models on a validation set, or they could be learned as part of the training process. In some cases, all models might be given equal weight.
The goal of a weighted ensemble is to produce a final prediction that is more accurate or robust than any individual model. This is particularly useful when the individual models have complementary strengths and weaknesses.

## Examples

### CLI Usage

Configuration template for the weighted ensemble algorithm:

```yaml title="config/method/ensemble/weighted_ensemble.yaml"
--8<-- "config/method/ensemble/weighted_ensemble.yaml"
```

Construct a weighted ensemble using our CLI tool `fusion_bench`:

```bash
# With specific weights, override via method.weights
fusion_bench method=ensemble/weighted_ensemble \
    method.weights=[0.3, 0.7] \
  modelpool=... \
  taskpool=...

# With equal weights (default)
fusion_bench method=ensemble/weighted_ensemble \
  modelpool=... \
  taskpool=...
```

### API Usage

The following Python code snippet demonstrates how to use the `WeightedEnsembleAlgorithm` class from the `fusion_bench.method` module to create a weighted ensemble of PyTorch models.

```python
from fusion_bench.method import WeightedEnsembleAlgorithm

# Instantiate the algorithm
algorithm = WeightedEnsembleAlgorithm(weights=[0.3, 0.7], normalize=True)

# Assume we have a modelpool or a list of PyTorch models (nn.Module instances) that we want to ensemble.
models = [...]  # List of nn.Module instances

# Run the algorithm on the modelpool or models.
ensemble_model = algorithm.run(modelpool)  # or algorithm.run(models)
```

Here's a step-by-step explanation:

1. **Instantiate the [`WeightedEnsembleAlgorithm`][fusion_bench.method.WeightedEnsembleAlgorithm]**: 
    - The algorithm is instantiated with two parameters: `weights` (a list of floats representing the weights for each model) and `normalize` (whether to normalize the weights).
    - If `weights` is set to `None`, the algorithm will automatically assign equal weights to all models.

2. **Prepare your models**: 
    - You can either use a `BaseModelPool` instance that contains your models, or directly provide a list of PyTorch `nn.Module` instances.
    - If you provide a list of models directly, the algorithm will automatically wrap them in a `BaseModelPool`.

3. **Run the algorithm**: 
    - The `run` method processes the modelpool and returns a `WeightedEnsembleModule` that represents the weighted ensemble of the input models.

Here we list the options for the weighted ensemble algorithm:

| Option      | Default | Description                                                                |
| ----------- | ------- | -------------------------------------------------------------------------- |
| `weights`   | `null`  | A list of floats representing the weights for each model in the ensemble. If `null`, equal weights are automatically assigned to all models. |
| `normalize` | `True`  | Whether to normalize the weights so that they sum to 1. Default is `True`. |

If `normalize` is set to `True`, the weights will be normalized so that they sum to 1. Mathematically, this means that the weights $w_i$ will be divided by the sum of all weights, so that

$$
F(x) = \frac{w_1}{\sum_{i=1}^n w_i} f_1(x) + \frac{w_2}{\sum_{i=1}^n w_i} f_2(x) + ... + \frac{w_n}{\sum_{i=1}^n w_i} f_n(x)
$$

When `weights` is set to `null` (or `None` in Python), the algorithm automatically assigns equal weights to all models: $w_i = \frac{1}{n}$ where $n$ is the number of models.


## Implementation Details

- [fusion_bench.method.WeightedEnsembleAlgorithm][]

