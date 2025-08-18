# Simple Averaging

Simple averaging, also known as isotropic merging or ModelSoups, aims to yield a more robust and generalizable model by combining multiple models of the same architecture.

Simple averaging is a technique frequently employed when there are multiple models that have been fine-tuned or independently trained from scratch.
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

### CLI Usage

Configuration template for the standard Simple Averaging algorithm:

```yaml title="config/method/simple_average.yaml"
--8<-- "config/method/simple_average.yaml"
```

Use the following command to run the standard Simple Averaging algorithm:

```bash
fusion_bench method=simple_average ...
```

###  API Usage

#### Algorithm Class

In this example, we demonstrate how to use the [`SimpleAverageAlgorithm`][fusion_bench.method.simple_average.SimpleAverageAlgorithm] class:

```python
from fusion_bench.method.simple_average import SimpleAverageAlgorithm

# Instantiate the SimpleAverageAlgorithm
algorithm = SimpleAverageAlgorithm()

# Assume we have a model pool with multiple models of the same architecture
modelpool = ...  # BaseModelPool instance

# Run the algorithm on the model pool
# Returns a new model with averaged parameters
merged_model = algorithm.run(modelpool)
```

#### Low-level Function Usage

You can also use the low-level [`simple_average`][fusion_bench.method.simple_average.simple_average] function directly:

```python
from fusion_bench.method.simple_average import simple_average

# For a list of models
models = [model1, model2, model3]
averaged_model = simple_average(models)

# For a list of state dictionaries
state_dicts = [model1.state_dict(), model2.state_dict(), model3.state_dict()]
averaged_state_dict = simple_average(state_dicts)
```

## Variants

### Standard Simple Averaging

The basic implementation (`SimpleAverageAlgorithm`) directly averages model parameters without any modifications.

### DARE Simple Averaging

A variant that incorporates DARE (Drop And REscale) techniques for improved performance.

- **Sparsity-aware merging**: Applies random dropping to parameters before averaging
- **Rescaling**: Optionally rescales remaining parameters after dropping to maintain magnitude

The DARE variant is particularly useful when dealing with fine-tuned models that may have redundant or conflicting parameters.

## Implementation Details

- [fusion_bench.method.simple_average.SimpleAverageAlgorithm][]
- [fusion_bench.method.simple_average.simple_average][]
