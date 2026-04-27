# Model Stock

Model Stock is a parameter-space merging method that leverages the geometric relationship between fine-tuned models. Rather than using fixed weights, it computes an interpolation ratio for each parameter group based on the angle between fine-tuned models relative to a pretrained anchor. The intuition is that when two fine-tuned models move in similar directions from the pretrained model, they can be merged more aggressively; when they diverge, the pretrained model acts as a stabilizer[^1].

<figure markdown="span">
  ![Model Stock](images/model_stock.png){ width="500" }
  <figcaption>Model Stock: Angle-based interpolation between fine-tuned models and pretrained anchor. Credit to <sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></figcaption>
</figure>

**Core Idea**. Given a pretrained model $\theta_0$ and fine-tuned models $\theta_1, \theta_2, \dots, \theta_K$, Model Stock computes the angle between the fine-tuned models' displacement vectors in parameter space. The interpolation ratio $t$ is then derived from this angle, yielding a merge:

$$\theta_{merged} = t \cdot \bar{\theta}_{ft} + (1 - t) \cdot \theta_0$$

where $\bar{\theta}_{ft}$ is the average of the fine-tuned models, and $t$ is computed from the cosine of the angles between pairs of fine-tuned models.

## Mathematical Formulation

### Two-Model Case

For two fine-tuned models $\theta_1$ and $\theta_2$, the angle between their displacement vectors relative to $\theta_0$ is computed per-parameter-group:

$$\cos \phi = \frac{(\theta_1 - \theta_0) \cdot (\theta_2 - \theta_0)}{\|\theta_1 - \theta_0\| \cdot \|\theta_2 - \theta_0\|}$$

The interpolation ratio is then:

$$t = \frac{K \cos \phi}{(K-1) \cos \phi + 1}$$

where $K = 2$ for the two-model case. The merged model is:

$$\theta_{merged} = t \cdot \frac{\theta_1 + \theta_2}{2} + (1 - t) \cdot \theta_0$$

When $\cos \phi$ is close to 1 (the models moved in the same direction), $t \approx 1$ and the merge relies heavily on the fine-tuned models. When $\cos \phi$ is near 0 or negative (the models diverged), $t$ shrinks, and the pretrained anchor contributes more.

### N-Model Case

For $K > 2$ fine-tuned models, Model Stock computes the average angle across all pairs:

1. Compute pairwise angles for all $\binom{K}{2}$ pairs.
2. Average the angles per parameter group.
3. Apply the same ratio formula using the averaged angle and $K$.
4. Average the fine-tuned models: $\bar{\theta}_{ft} = \frac{1}{K} \sum_{i=1}^{K} \theta_i$.
5. Merge: $\theta_{merged} = t \cdot \bar{\theta}_{ft} + (1 - t) \cdot \theta_0$.

### Angle Computation

The `compute_angle()` function calculates the angle between two state dictionaries relative to a reference:

```python
def compute_angle(state_dict_1, state_dict_2, ref_state_dict, ignore_keys=[], return_cos=False):
    for key in ref_state_dict:
        vector1 = (state_dict_1[key] - ref_state_dict[key]).clone().detach()
        vector2 = (state_dict_2[key] - ref_state_dict[key]).clone().detach()
        cosine_val = sum(vector1 * vector2) / (sqrt(sum(vector1^2) * sum(vector2^2)) + EPS)
        angle = acos(cosine_val)  # in radians
    return angle_dict
```

### Ratio Computation

```python
def compute_ratio(angle_dict, k=2):
    for key in angle_dict:
        angle = rad2deg(angle_dict[key])
        ratio = k * cos(angle) / ((k - 1) * cos(angle) + 1 + EPS)
    return ratio_dict
```

### Weight Merge

```python
def merge_weights(w1, w2, w0, ratio):
    w12 = (w1 + w2) / 2  # average of fine-tuned models
    for key, r in ratio.items():
        w_merge[key] = w12[key] * r + w0[key] * (1.0 - r)
    return w_merge
```

## Examples

### CLI Usage

Configuration template:

```yaml title="config/method/model_stock/model_stock.yaml"
--8<-- "config/method/model_stock/model_stock.yaml"
```

Run Model Stock:

```bash
fusion_bench method=model_stock/model_stock \
    method.model_save_path=outputs/model_stock_merged \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_two_tasks \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_two_tasks
```

The default configuration ignores certain CLIP-specific parameters that are not fine-tuned:

```yaml
ignore_keys:
  - "model.positional_embedding"
  - "model.text_projection"
  - "model.logit_scale"
  - "model.token_embedding.weight"
  - "model.ln_final.weight"
  - "model.ln_final.bias"
```

### API Usage

```python
from fusion_bench.method.model_stock import ModelStock

# Instantiate the algorithm
algorithm = ModelStock(
    ignore_keys=[],
    model_save_path="outputs/merged",
)

# Run the algorithm on a model pool
# The model pool must contain a pretrained model (_pretrained_) and at least 2 fine-tuned models
merged_model = algorithm.run(modelpool)
```

## Implementation Details

### Algorithm Steps

The `ModelStock.run()` method follows these steps:

1. **Load Models**: Load the pretrained model and all fine-tuned models from the model pool.
2. **Compute Angles**:
   - For 2 models: compute pairwise angles directly.
   - For $N$ models: compute all pairwise angles and average them.
3. **Compute Ratios**: Convert angles to interpolation ratios using the `compute_ratio()` formula.
4. **Merge Weights**:
   - For 2 models: apply `merge_weights()` directly.
   - For $N$ models: average the fine-tuned models, then apply the ratio formula.
5. **Save**: If `model_save_path` is set, save the merged model and generate a model card.

### Key Features

- **Per-Parameter-Group Ratios**: Each parameter (e.g., each layer's weight matrix) gets its own interpolation ratio, allowing fine-grained control.
- **Ignore Keys**: Parameters that were not fine-tuned (e.g., positional embeddings) can be excluded from angle computation.
- **Shape Validation**: Parameters with mismatched shapes are skipped with a warning.
- **LazyStateDict Support**: Works with FusionBench's lazy state dict for memory-efficient merging of large models.
- **Model Card Generation**: Automatically generates a README model card when saving.

### Model Stock Class

```python
@auto_register_config
class ModelStock(SimpleProfilerMixin, BaseAlgorithm):
    def __init__(
        self,
        ignore_keys: Optional[List[str]] = None,
        model_save_path: Optional[str] = None,
        model_save_kwargs: Optional[DictConfig] = None,
    ):
        ...
```

The `@auto_register_config` decorator automatically maps constructor arguments to YAML config keys.

## Hyperparameter Guidelines

- **`ignore_keys`**: Exclude parameters that were frozen during fine-tuning. For CLIP vision models, this typically includes positional embeddings, text projection, logit scale, and text token embeddings.
- **`model_save_path`**: Set to save the merged model. Supports both standard `nn.Module` and HuggingFace `PreTrainedModel` saving.

## Implementation Details

- [ModelStock][fusion_bench.method.model_stock.ModelStock]

[^1]: (2024) Model Stock: All we need is just a few fine-tuned models. https://arxiv.org/abs/2403.19522
