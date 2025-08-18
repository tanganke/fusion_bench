# Simple Ensemble

Ensemble methods are simple and effective ways to improve the performance of machine learning models. 
These methods combine the outputs of multiple models to create a stronger model. 
A simple ensemble takes the average of the predictions from multiple models without any weighting.

Formally, given a set of $n$ models, each model $f_i$ produces a prediction $f_i(x)$ for an input $x$. The final prediction $F(x)$ of the simple ensemble is the unweighted average of the individual model predictions:

$$
F(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)
$$

This approach assumes that all models contribute equally to the final prediction and is particularly effective when the individual models have similar performance levels. 


## Examples

### CLI Usage

Configuration template for the ensemble algorithm:

```yaml title="config/method/ensemble/simple_ensemble.yaml"
--8<-- "config/method/ensemble/simple_ensemble.yaml"
```

create a simple ensemble of CLIP-ViT models for image classification tasks.

```bash
fusion_bench \
  method=ensemble/simple_ensemble \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 
```

### API Usage

The following Python code snippet demonstrates how to use the [`SimpleEnsembleAlgorithm`][fusion_bench.method.SimpleEnsembleAlgorithm] class from the `fusion_bench.method` module to create a simple ensemble of PyTorch models.

```python
from fusion_bench.method import SimpleEnsembleAlgorithm

# Instantiate the SimpleEnsembleAlgorithm
algorithm = SimpleEnsembleAlgorithm()

# Assume we have a list of PyTorch models (nn.Module instances) or a modelpool that we want to ensemble.
models = [...]

# Run the algorithm on the modelpool or models.
ensemble_model = algorithm.run(modelpool)  # or algorithm.run(models)
```

Here's a step-by-step explanation:

1. **Instantiate the [`SimpleEnsembleAlgorithm`][fusion_bench.method.SimpleEnsembleAlgorithm]**: 
    - The algorithm requires no parameters for initialization since it uses equal weights for all models.

2. **Prepare your models**: 
    - You can either use a [`BaseModelPool`][fusion_bench.BaseModelPool] instance that contains your models, or directly provide a list of PyTorch `nn.Module` instances.
    - The algorithm will load models from the modelpool using `modelpool.load_model()` for each model name.

3. **Run the algorithm**: 
    - The `run` method processes the modelpool and returns an `EnsembleModule` that represents the simple ensemble of the input models.
    - The resulting ensemble computes the average of all model predictions.


## Implementation Details

- [fusion_bench.method.SimpleEnsembleAlgorithm][]
