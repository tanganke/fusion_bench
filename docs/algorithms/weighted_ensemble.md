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

The following Python code snippet demonstrates how to use the `WeightedEnsembleAlgorithm` class from the `fusion_bench.method` module to create a weighted ensemble of PyTorch models.

```python
from omegaconf import DictConfig
from fusion_bench.method import WeightedEnsembleAlgorithm

#Instantiate the algorithm
method_config = {'name': 'weighted_ensemble', 'weights': [0.3, 0.7]}
algorithm = WeightedEnsembleAlgorithm(DictConfig(method_config))

# Assume we have a list of PyTorch models (nn.Module instances) that we want to ensemble.
models = [...]

# Run the algorithm on the models.
merged_model = algorithm.run(models)
```

Here's a step-by-step explanation:

1. Instantiate the `WeightedEnsembleAlgorithm`:
    - A dictionary `method_config` is created with two keys: `'name'` and `'weights'`. The `'name'` key is set to `'weighted_ensemble'` indicating the type of ensemble method to use. The `'weights'` key is set to a list of weights `[0.3, 0.7]` indicating the weights assigned to each model in the ensemble.
    - The `method_config` dictionary is converted to a `DictConfig` object, which is a configuration object used by the `omegaconf` library.
    - The `WeightedEnsembleAlgorithm` is then instantiated with the `DictConfig` object as an argument.

2. Assume a list of PyTorch models that you want to ensemble. This list is assigned to the variable `models`. The actual models are not shown in this code snippet.

3. Run the algorithm on the models: The `run` method of the `WeightedEnsembleAlgorithm` instance is called with the `models` list as an argument. The result is a merged model that represents the weighted ensemble of the input models. This merged model is assigned to the variable `merged_model`.

Here we list the options for the weighted ensemble algorithm:

| Option      | Default | Description                                                                |
| ----------- | ------- | -------------------------------------------------------------------------- |
| `weights`   |         | A list of floats representing the weights for each model in the ensemble.  |
| `normalize` | `True`  | Whether to normalize the weights so that they sum to 1. Default is `True`. |

if `normalize` is set to `True`, the weights will be normalized so that they sum to 1.  Mathematically, this means that the weights $w_i$ will be divided by the sum of all weights, so that

$$
F(x) = \frac{w_1}{\sum_{i=1}^n w_i} f_1(x) + \frac{w_2}{\sum_{i=1}^n w_i} f_2(x) + ... + \frac{w_n}{\sum_{i=1}^n w_i} f_n(x)
$$

## Code Intergration

Configuration template for the weighted ensemble algorithm:

```yaml title="config/method.weighted_ensemble.yaml"
name: weighted_ensemble

# this should be a list of floats, one for each model in the ensemble
# If weights is null, the ensemble will use the default weights, which are equal weights for all models.
weights: null
nomalize: true
```

Construct a weighted ensemble using our CLI tool `fusion_bench`:

```bash
fusion_bench method=weighted_ensemble \
    method.weights=[0.3, 0.7] \
  modelpool=... \
  taskpool=...
```

## References

::: fusion_bench.method.WeightedEnsembleAlgorithm
