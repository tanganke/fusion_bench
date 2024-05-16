# Simple Averaging

Simple averaging is known in the literature as ModelSoups, aims to yield a more robust and generalizable model. 

In the context of full fine-tuned models, the weights are averaged directly. Concretely, this means that if we have $n$ models with their respective weights $\theta_i$, the weights of the final model $\theta$ are computed as:

$$ \theta = \frac{1}{n} \sum_{i=1}^{n} \theta_i $$


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

::: fusion_bench.method.SimpleAverageAlgorithm
    options:
        members: true
