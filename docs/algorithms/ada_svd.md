# AdaSVD (Adaptive SVD-based Merging)

AdaSVD is an adaptive model merging algorithm specifically designed for CLIP vision encoders. It combines ideas from Sparse Mixture of Low-Rank Experts (SMILE) with data-driven routing weight computation to produce a merged model that adaptively combines multiple fine-tuned experts. The algorithm first upscaling the pretrained model into a Mixture-of-Experts (MoE) architecture, then reduces the MoE back to dense linear layers by computing average routing weights over a budget of calibration samples.

## Algorithm Overview

### SMILE Upscaling

The algorithm begins by replacing each linear layer in the CLIP vision encoder with a `SmileMoELinear` module. Each MoE module consists of:

- **The pretrained linear layer**: Serves as the base/expert 0.
- **Expert linear layers**: One per fine-tuned model, each containing the corresponding layer's parameters from a fine-tuned model.
- **A gating network**: Routes input tokens to a subset of experts based on SVD-derived routing scores.

The gating uses the difference between expert weights and the pretrained weight ($\Delta W = W_{\text{expert}} - W_{\text{pretrained}}$) to compute routing scores. The `gate_k` parameter controls the top-k selection.

### Routing Weight Accumulation

After upscaling, the algorithm performs a forward pass through the model on a set of calibration samples (controlled by `num_samples`). For each layer, hooks accumulate the routing weights produced by the gating network:

$$\bar{w}_i = \frac{1}{N} \sum_{n=1}^{N} \text{softmax}(g_i(x_n))_i$$

where $g_i(x_n)$ is the gate's output for expert $i$ on sample $x_n$.

### MoE Reduction

Once routing weights are accumulated, each MoE linear layer is reduced back to a standard linear layer via weighted averaging:

$$W_{\text{merged}} = W_{\text{pretrained}} + \sum_{i=1}^{M} \bar{w}_i \cdot (W_i - W_{\text{pretrained}})$$

This is equivalent to a weighted average where the pretrained model has weight 1 and each expert $i$ has weight $\bar{w}_i$ (optionally scaled by `scaling_factor`).

### Non-Linear Module Handling

For non-linear modules (e.g., layer norms), the `average_experts` flag controls behavior:

- **`true`**: Average the parameters of all expert models.
- **`false`**: Keep the pretrained model's parameters.

## Mathematical Formulation

### Upscaling

For each linear layer $l$ with pretrained weight $W_0^{(l)}$ and fine-tuned weights $\{W_i^{(l)}\}_{i=1}^{M}$:

1. Construct a `SmileMoELinear` module with experts $\{W_0^{(l)}, W_1^{(l)}, ..., W_M^{(l)}\}$.
2. The gate computes routing scores using the SVD of the weight differences.
3. The `top_k` parameter is set to $M$ (all experts), with dense routing.

### Routing Weight Computation

For input hidden states $h$ passed through layer $l$:

$$s_i = g_i(h) = \text{score of expert } i$$

$$w_i = \text{softmax}(s)_i$$

Over $N$ calibration samples:

$$\bar{w}_i = \frac{1}{N} \sum_{n=1}^{N} w_i^{(n)}$$

### Reduction (MoE to Dense)

The final merged weight for layer $l$ is computed as a weighted combination:

$$W_{\text{merged}}^{(l)} = W_0^{(l)} \cdot 1 + \sum_{i=1}^{M} W_i^{(l)} \cdot (\bar{w}_i \cdot \alpha)$$

where $\alpha$ is the optional `scaling_factor`. When `scaling_factor` is a list, it provides per-expert scaling. The `WeightedAverageAlgorithm` with `normalize=False` is used, meaning weights are not renormalized.

### Hidden State Propagation

The algorithm processes the CLIP transformer layer by layer:

1. Extract hidden states from the input embeddings through the pre-layernorm.
2. For each encoder layer, propagate hidden states through all MoE modules.
3. Accumulate routing weights at each layer.
4. Use the output hidden states as input to the next layer.

## Configuration

```yaml title="config/method/ada_svd/clip_vision.yaml"
--8<-- "config/method/ada_svd/clip_vision.yaml"
```

Key configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `scaling_factor` | Scaling for expert weights (float or list) | `null` |
| `num_samples` | Number of calibration samples per dataset | `256` |
| `gate_k` | Top-k experts for routing | `16` |
| `average_experts` | Average non-linear expert modules | `false` |
| `device` | Device for computation: `cuda` or `cpu` | `cuda` |
| `upscaling_accelerator` | Accelerator for SMILE upscaling | `null` |
| `seed` | Random seed for reproducibility | `0` |

## Examples

### CLI Usage

```bash
fusion_bench \
    method=ada_svd/clip_vision \
    method.num_samples=256 \
    method.gate_k=16 \
    method.scaling_factor=1.0 \
    method.average_experts=false \
    method.device=cuda \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method.ada_svd.clip_vision import AdaSVDMergingForCLIPVisionModel

algorithm = AdaSVDMergingForCLIPVisionModel(
    scaling_factor=None,
    num_samples=256,
    gate_k=16,
    average_experts=False,
    device="cuda",
    upscaling_accelerator=None,
    seed=0,
)

merged_model = algorithm.run(modelpool)
```

## Implementation Details

### Data Preparation

The `prepare_data` method samples `num_samples` from each training dataset in the model pool. It uses `random_split` to select a subset, then wraps them in `CLIPDataset` with the model pool's processor. All datasets are concatenated into a single `ConcatDataset`.

### Model Preparation

The `prepare_model` method loads the pretrained model and all fine-tuned models, then calls `merge` to perform SMILE upscaling. Models are moved to the specified device (GPU if `cuda` and available).

### Upscaling Process

- **Linear layers**: Each `nn.Linear` module is replaced with a `SmileMoELinear` using the pretrained and expert weights. The `k=-1` setting enables dense experts, and `routing_use_diff=True` uses the weight difference for routing.
- **Leaf modules**: When `average_experts=True`, leaf modules (modules with no sub-modules) are averaged across experts using `simple_average`.

### Routing Weight Collection

The `AvgRoutingWeightsMetric` class registers as a forward hook on each `SmileMoELinear`. It computes the softmax routing weights for each forward pass and accumulates them. After processing all samples, `compute()` returns the average routing weights.

### Layer-by-Layer Processing

The algorithm processes transformer layers sequentially. Hidden states are propagated layer by layer, with routing weights accumulated at each layer. This is more memory-efficient than processing the entire network at once.

### Memory Efficiency

After upscaling a linear layer, the original modules in the fine-tuned models are set to `None` to free memory. The algorithm also clears the fine-tuned model references progressively.

## References

[^1]: (arXiv 2024) SMILE: Zero-Shot Sparse Mixture of Low-Rank Experts Construction From Pre-Trained Foundation Models. http://arxiv.org/abs/2408.10174. Introduces the SMILE upscaling technique that AdaSVD builds upon.
