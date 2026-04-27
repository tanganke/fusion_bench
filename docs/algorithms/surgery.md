# Surgery for Model Merging

After merging multiple task-specific models, the merged model often exhibits a representation gap: the internal features it produces for task-specific inputs differ from what the individual fine-tuned models would produce. Surgery addresses this by optimizing a set of learnable blending parameters that adjust the merged model's feature representations to better align with the original task-specific models.

**The Core Idea**. Traditional merging methods (e.g., AdaMerging, Task Arithmetic) operate in the weight space, computing a single set of merged weights. However, optimal merging may require different interpolation strategies at the representation level. Surgery introduces learnable alpha parameters that reweight the contributions of each expert model when computing features, and these alphas are optimized via gradient descent to minimize the L1 distance between the merged model's features and the target fine-tuned model's features.

**The Surgery Model Wrapper**. The algorithm wraps the merged model in a `SurgeryModelWrapper` that introduces learnable parameters per layer and per task. For each input sample from task $t$, the wrapper computes features as a weighted combination of features from each expert model and the merged model:

$$h_{\text{surgery}}(x_t) = \sum_{i} \alpha_i^{(t)} \cdot h_i(x_t)$$

where $h_i(x_t)$ is the feature produced by the $i$-th expert model on input $x_t$ from task $t$, and $\alpha_i^{(t)}$ are the learnable surgery parameters.

**Optimization**. The surgery parameters are optimized to minimize the feature-level reconstruction loss:

$$\mathcal{L} = \| h_{\text{surgery}}(x_t) - h_{\text{target}}(x_t) \|_1$$

where $h_{\text{target}}(x_t)$ is the feature produced by the fine-tuned model for task $t$. The optimization runs for `surgery_steps` iterations using the Adam optimizer with a fixed learning rate of $10^{-3}$.

## Two-Phase Pipeline

The Surgery implementation in FusionBench follows a two-phase approach:

1. **AdaMerging Phase**: First, layer-wise AdaMerging is performed to obtain initial merge weights via test-time adaptation. This produces a statically merged model.

2. **Surgery Phase**: The merged model is then wrapped in a `SurgeryModelWrapper`, and the learnable alpha parameters are optimized over task-specific calibration data to reduce the representation gap.

## Examples

### CLI Usage

```yaml title="config/method/surgery/adamerging_surgery.yaml"
--8<-- "config/method/surgery/adamerging_surgery.yaml"
```

```bash
fusion_bench \
  method=surgery/adamerging_surgery \
  method.surgery_steps=1000 \
  method.eval_iterations=200 \
  method.max_steps=1000 \
  method.lr=0.001 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `surgery_steps` | int | 1000 | Number of gradient descent steps for surgery optimization. |
| `eval_iterations` | int | 200 | Frequency of evaluation during surgery training. |
| `max_steps` | int | 1000 | Max steps for the initial AdaMerging test-time adaptation. |
| `lr` | float | 1e-3 | Learning rate for the surgery Adam optimizer. |
| `batch_size` | int | 16 | Batch size for calibration data during both phases. |
| `num_workers` | int | 8 | Number of DataLoader workers. |
| `weights` | str/None | null | Path to pre-computed merge weights. If set, skips test-time adaptation. |
| `clamp_weights` | bool | false | Whether to clamp AdaMerging weights to [0, 1]. |
| `save_merging_weights` | str | 'merging_weights.pt' | Path to save the AdaMerging merge weights. |

## Output

The algorithm returns a dictionary with two keys:

- `"adamerging"`: The statically merged model from the AdaMerging phase.
- `"surgery"`: The SurgeryModelWrapper with learnable alpha parameters.

## Implementation Details

- [fusion_bench.method.surgery.clip_layer_wise_adamerging_surgery.CLIPLayerWiseAdaMergingSurgeryAlgorithm][]
- [fusion_bench.models.surgery.surgerymodelwrapper.SurgeryModelWrapper][]

[^1]: (ICLR 2024) AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575
[^2]: (ICML 2024) Representation Surgery for Multi-Task Model Merging. http://arxiv.org/abs/2402.02705
