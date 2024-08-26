# Simple Averaging

Simple averaging is known in the literature as isotropic merging, ModelSoups, aims to yield a more robust and generalizable model.
Simple Averaging is a technique frequently employed when there are multiple models that have been fine-tuned or independently trained from scratch. 
Specifically, if we possess $n$ models that share a common architecture but different weights denoted as $\theta_i$, the weights of the merged model, represented as $\theta$, are computed as follows:

$$ \theta = \frac{1}{n} \sum_{i=1}^{n} \theta_i $$

This equation simply states that each weight of the final model is the average of the corresponding weights in the individual models. For example, if we have three models and the weight of the first neuron in the first layer is 0.1, 0.2, and 0.3 in each model respectively, the weight of that neuron in the final model will be (0.1 + 0.2 + 0.3) / 3 = 0.2.

Simple averaging is a straightforward and scalable method for model fusion. It does not require any additional training or fine-tuning, making it a good choice when computational resources are limited, where maintaining an ensemble of models is not feasible.

This method often assumes that all models are equally good. 
If some models are significantly better than others, it might be beneficial to assign more weight to the better models when averaging. 
This can be done by using weighted averaging, where each model's contribution to the final model is weighted by its performance on a validation set or some other metric.
See [Weighed Averaging](weighted_averaging.md) for more details.
Otherwise, the poor model may have a negative impact on the merged model.


## Examples

In this example, we will demonstrate how to use the `SimpleAverageAlgorithm` class from the `fusion_bench.method` module. 
This algorithm is used to merge multiple models by averaging their parameters.

```python
from fusion_bench.method.simple_average import SimpleAverageAlgorithm

# Instantiate the SimpleAverageAlgorithm
# This algorithm will be used to merge multiple models by averaging their parameters.
algorithm = SimpleAverageAlgorithm()

# Assume we have a list of PyTorch models (nn.Module instances) that we want to merge.
# The models should all have the same architecture.
models = [...]

# Run the algorithm on the models.
# This will return a new model that is the result of averaging the parameters of the input models.
merged_model = algorithm.run(models)
```

The `run` method of the `SimpleAverageAlgorithm` class takes a list of models as input and returns a new model. 
The new model's parameters are the average of the parameters of the input models. 
This is useful in scenarios where you have trained multiple models and want to combine them into a single model that hopefully performs better than any individual model.

## Code Integration

Configuration template for the Simple Averaging algorithm:

```yaml title="config/method/simple_average.yaml"
name: simple_average
```

use the following command to run the Simple Averaging algorithm:

```bash
fusion_bench method=simple_average ...
```

## References

::: fusion_bench.method.simple_average.SimpleAverageAlgorithm
    options:
        members: true
