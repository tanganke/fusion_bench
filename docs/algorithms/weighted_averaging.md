# Weighted Averaging

Weighted averaging, also known as weight-ensembling.
In the context of full fine-tuned models, the weights are averaged according to their respective performance weights. Concretely, this means that if we have $n$ models with their respective weights $\theta_i$ and model-wise weights $w_i$, the weights of the final model $\theta$ are computed as:

$$ \theta = \sum_{i=1}^{n} w_i \theta_i $$

## Code Integration

Configuration template for the Weighted Averaging algorithm:

```yaml title="config/method/weighted_average.yaml"
name: weighted_average
normalize: true # if true, the weights will be normalized before merging
weights: # List of weights for each model
  - 0.5
  - 0.5
```

Use the following command to run the Weighted Averaging algorithm:

```bash
fusion_bench method=weighted_average ...
```

## References

::: fusion_bench.method.WeightedAverageAlgorithm
    options:
        members: true
