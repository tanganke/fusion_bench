# Weighted Averaging

Weighted averaging, also known as weight-ensembling, combines multiple models by averaging their parameters according to specified weights. This approach allows for non-uniform combination of models, where better-performing or more reliable models can be given higher weights in the final merged model.

In the context of model fusion, if we have $n$ models with their respective parameters $\theta_i$ and model-wise weights $w_i$, the parameters of the final merged model $\theta$ are computed as:

$$ \theta = \sum_{i=1}^{n} w_i \theta_i $$

where the weights $w_i$ can optionally be normalized to sum to 1.

## Examples

### CLI Usage

#### General Pytorch Models

The [`WeightedAverageAlgorithm`][fusion_bench.method.WeightedAverageAlgorithm] works with general PyTorch models and performs weighted averaging of all model parameters.

Configuration template for the standard Weighted Averaging algorithm:

```yaml title="config/method/linear/weighted_average.yaml"
--8<-- "config/method/linear/weighted_average.yaml"
```

Use the following command to run the Weighted Averaging algorithm:

```bash
fusion_bench method=linear/weighted_average ...
```

The following command merges eight CLIP-ViT models using a weighted average approach:

```bash
# Note: Since `method.normalize=true`, the weights are normalized to sum to 1, making this example equivalent to simple averaging.
fusion_bench \
    method=linear/weighted_average \
    method.normalize=true \
    method.weights=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3] \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

#### Large Language Models

The [`WeightedAverageForLLama`][fusion_bench.method.WeightedAverageForLLama] is specialized for large language models with additional features:

- Backbone-only merging option
- Model saving capabilities
- Hub integration support

Configuration template for LLaMA/Mistral model weighted averaging:

```yaml title="config/method/linear/weighted_average_for_llama.yaml"
--8<-- "config/method/linear/weighted_average_for_llama.yaml"
```

Use the following command:

```bash
fusion_bench method=linear/weighted_average_for_llama ...
```


Example of merging LLaMA models with different weights:

```bash
fusion_bench \
    method=linear/weighted_average_for_llama \
    method.weights=[0.3,0.7] \
    method.normalize=true \
    method.backbone_only=true \
    method.merged_model_save_path=outputs/merged_llama_model \
    modelpool=CausalLMPool/simle_mixtral_exp_v4.yaml \
    taskpool=dummy
```

### API Usage

#### General Pytorch Models

```python
from fusion_bench.method.weighted_average import WeightedAverageAlgorithm

# Create the algorithm with custom weights
algorithm = WeightedAverageAlgorithm(
    normalize=True,  # Normalize weights to sum to 1
    weights=[0.3, 0.5, 0.2],  # Custom weights for 3 models
    verbose=True
)

# Run on a model pool
merged_model = algorithm.run(modelpool)
```

#### Large Language Models

```python
from fusion_bench.method import WeightedAverageForLLama

# Create the algorithm for LLaMA models
algorithm = WeightedAverageForLLama(
    normalize=True,
    weights=[0.4, 0.6],
    backbone_only=True,  # Only merge backbone, keep heads
    merged_model_save_path="./merged_model",
    save_tokenizer=True,
    push_to_hub=False
)

# Run on a CausalLMPool
merged_model = algorithm.run(causal_lm_pool)
```

## Implementation Details

- [fusion_bench.method.weighted_average.weighted_average.WeightedAverageAlgorithm][]
- [fusion_bench.method.weighted_average.llama.WeightedAverageForLLama][]

