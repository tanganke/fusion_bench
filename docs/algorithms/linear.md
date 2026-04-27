# Linear Model Merging Methods

Linear model merging encompasses a family of methods that combine model parameters through linear operations -- interpolation, extrapolation, and weighted averaging. These methods form the foundation of model fusion, with more advanced techniques often building upon them.

## Overview

Linear merging methods operate by computing a linear combination of model parameters. Given models with parameters $\theta_1, \theta_2, \dots, \theta_K$, the merged model is:

$$\theta_{merged} = \sum_{i=1}^{K} w_i \theta_i$$

where $\sum_i w_i = 1$ and $w_i$ are the merging weights. The specific choice of weights and the relationship between the models define the variant.

FusionBench implements several linear merging methods:

1. **Linear Interpolation**: Interpolates between two models with a parameter $t \in [0, 1]$.
2. **ExPO (Extrapolation)**: Extrapolates from a pretrained model through a fine-tuned model.
3. **ExPO for LLaMA**: A LLaMA-specific variant of ExPO with layer-wise control.
4. **Simple Average for Causal LM**: Uniform averaging with optional backbone-only merging.

## Linear Interpolation

The simplest linear merge method interpolates between two models:

$$\theta = (1 - t) \theta_1 + t \theta_2$$

where $t \in [0, 1]$ controls the interpolation: $t = 0$ yields $\theta_1$, $t = 1$ yields $\theta_2$, and $t = 0.5$ gives the simple average.

```python
from fusion_bench.method import LinearInterpolationAlgorithm

algorithm = LinearInterpolationAlgorithm(t=0.5)
merged_model = algorithm.run(modelpool)  # modelpool must have exactly 2 models
```

### Configuration

```yaml title="config/method/linear/linear_interpolation.yaml"
--8<-- "config/method/linear/linear_interpolation.yaml"
```

### CLI Usage

```bash
fusion_bench method=linear/linear_interpolation \
    method.t=0.5 \
    modelpool=CausalLMPool/two_models \
    taskpool=...
```

## ExPO (Extrapolation)

ExPO (Extrapolation from Pretrained to Optimized) extends the idea of linear interpolation into extrapolation. Given a pretrained (SFT) model $\theta_{pre}$ and a fine-tuned (RLHF) model $\theta_{ft}$, ExPO computes:

$$\theta_{merged} = \theta_{ft} + \alpha (\theta_{ft} - \theta_{pre})$$

where $\alpha$ is the extrapolation factor. When $\alpha > 0$, the merged model lies on the ray from $\theta_{pre}$ through $\theta_{ft}$, beyond $\theta_{ft}$. This can amplify the alignment improvements introduced by fine-tuning[^2].

### General ExPO

For general `nn.Module` models, the `ExPOAlgorithm` class handles any model architecture:

```python
from fusion_bench.method import ExPOAlgorithm

algorithm = ExPOAlgorithm(extrapolation_factor=0.1)
merged_model = algorithm.run(modelpool)
```

When multiple RLHF models are provided, ExPO first averages them via `SimpleAverageAlgorithm`, then extrapolates from the pretrained model through the averaged RLHF model.

### Configuration

```yaml title="config/method/linear/expo.yaml"
--8<-- "config/method/linear/expo.yaml"
```

### CLI Usage

```bash
fusion_bench method=linear/expo \
    method.extrapolation_factor=0.1 \
    modelpool=CausalLMPool/sft_and_rlhf \
    taskpool=...
```

## ExPO for LLaMA

The `ExPOAlgorithmForLlama` class provides fine-grained control over which parts of a LLaMA model are extrapolated. This is critical because different components (attention, MLP, embeddings, lm_head) may benefit from different treatment.

### Key Parameters

- **`extrapolation_factor`**: The extrapolation coefficient $\alpha$.
- **`attention_scaling_factor`**: Scales the extrapolation factor for attention layers separately. The effective factor for attention becomes `extrapolation_factor * attention_scaling_factor`.
- **`only_on_backbone`**: When `True`, only the backbone (transformer layers) is merged; the lm_head is kept from the RLHF model.
- **`on_linear_weights`** / **`on_linear_bias`**: Control whether linear weights and biases are extrapolated.
- **`on_embedding`**: Whether to extrapolate the token embedding layer.
- **`fix_first_n_layers`** / **`fix_last_n_layers`**: Skip extrapolation for the first/last N layers (supports `"half"` for half the layers).
- **`magnitude_sparsity_ratio`**: Optionally apply magnitude pruning to the delta vector before extrapolation.

### Mathematical Formulation

For each layer $l$, the LLaMA-specific ExPO applies:

$$\theta^{(l)}_{merged} = \theta^{(l)}_{ft} + \alpha_l (\theta^{(l)}_{ft} - \theta^{(l)}_{pre})$$

where $\alpha_l = \alpha \cdot \alpha_{attn}$ for attention layers and $\alpha_l = \alpha$ for MLP layers.

If `magnitude_sparsity_ratio` is set, the delta $\delta = \theta_{ft} - \theta_{pre}$ is first pruned via unstructured magnitude pruning before scaling.

### ExPO with DARE for LLaMA

The `ExPOWithDareForLLama` variant first merges the RLHF models using DARE simple averaging (random drop and rescale), then applies ExPO extrapolation. This combines the benefits of DARE's interference reduction with ExPO's extrapolation:

```python
from fusion_bench.method import ExPOWithDareForLLama

algorithm = ExPOWithDareForLLama(
    extrapolation_factor=0.1,
    dare_sparsity_ratio=0.5,
    dare_only_on_linear_weights=True,
    dare_rescale=True,
)
```

### Configuration

```yaml title="config/method/linear/llama_expo.yaml"
--8<-- "config/method/linear/llama_expo.yaml"
```

### CLI Usage

```bash
fusion_bench method=linear/llama_expo \
    method.extrapolation_factor=0.1 \
    method.attention_scaling_factor=1.0 \
    method.only_on_backbone=true \
    modelpool=CausalLMPool/sft_and_rlhf \
    taskpool=...
```

## Simple Average for Causal LM

The `SimpleAverageForCausalLM` class extends the basic simple average with Causal LM-specific features:

- **`merge_backbone`**: When `True`, only the backbone (transformer layers) is averaged. The lm_head is taken from the pretrained model. This is useful when merging models with different heads (e.g., chat vs. generation).
- **`model_save_path`**: Save the merged model and tokenizer to the specified path.
- **`show_pbar`**: Show a progress bar during merging.

### Configuration

```yaml title="config/method/linear/simple_average_for_causallm.yaml"
--8<-- "config/method/linear/simple_average_for_causallm.yaml"
```

### CLI Usage

```bash
fusion_bench method=linear/simple_average_for_causallm \
    method.merge_backbone=false \
    method.model_save_path=outputs/merged_model \
    method.show_pbar=true \
    modelpool=CausalLMPool/multiple_models \
    taskpool=...
```

### API Usage

```python
from fusion_bench.method import SimpleAverageForCausalLM

algorithm = SimpleAverageForCausalLM(
    merge_backbone=False,
    model_save_path="outputs/merged",
    show_pbar=True,
)
merged_model = algorithm.run(modelpool)
```

## Implementation Details

### ExPO Merge Function

The core `expo_merge()` function implements the extrapolation at the parameter level:

```python
def expo_merge(sft_model, rlhf_model, extrapolation_factor, inplace=True, enable_grad=False):
    for (sft_name, sft_param), (rlhf_name, rlhf_param) in zip(
        sft_model.named_parameters(), rlhf_model.named_parameters()
    ):
        rlhf_param.data = rlhf_param.data + extrapolation_factor * (
            rlhf_param.data - sft_param.data
        )
    return rlhf_model
```

### Linear Interpolation

The `LinearInterpolationAlgorithm` uses `state_dict_weighted_sum` to combine two state dictionaries:

```python
state_dict = state_dict_weighted_sum(
    [primary_state_dict, secondary_state_dict], [1 - self.t, self.t]
)
```

## Choosing a Method

| Scenario | Recommended Method |
|---|---|
| Two models, equal importance | Linear Interpolation (t=0.5) or Simple Average |
| Two models, unequal importance | Linear Interpolation with tuned $t$ |
| Pretrained + aligned model | ExPO (general or LLaMA) |
| Multiple RLHF models + SFT | ExPO (auto-averages RLHF models) |
| Multiple RLHF + SFT, large models | ExPO with DARE for LLaMA |
| Causal LMs with different heads | Simple Average for Causal LM (merge_backbone=True) |

## Implementation Details

- [ExPOAlgorithm][fusion_bench.method.ExPOAlgorithm]
- [LinearInterpolationAlgorithm][fusion_bench.method.LinearInterpolationAlgorithm]
- [ExPOAlgorithmForLlama][fusion_bench.method.ExPOAlgorithmForLlama]
- [ExPOWithDareForLLama][fusion_bench.method.ExPOWithDareForLLama]
- [SimpleAverageForCausalLM][fusion_bench.method.SimpleAverageForCausalLM]

[^2]: (2024) Zheng et al. Weak-to-Strong Extrapolation Expedites Alignment. https://arxiv.org/abs/2404.12717
