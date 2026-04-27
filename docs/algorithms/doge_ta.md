# DogeTA (DOGE Task Arithmetic)

DogeTA (DOGE Task Arithmetic) extends the Task Arithmetic framework by introducing a learnable delta vector that is optimized in a reduced subspace to improve the compatibility among task vectors. The algorithm models multi-task model merging as adaptive projective gradient descent, where task vectors are adjusted to minimize pairwise conflicts before being combined.

## Algorithm Overview

### Core Idea

Standard Task Arithmetic simply sums task vectors: $\theta = \theta_0 + \lambda \sum_i \tau_i$. However, task vectors from different tasks often conflict in certain parameter dimensions, leading to negative transfer. DogeTA addresses this by:

1. **Computing task vectors** for each fine-tuned model relative to the pretrained base.
2. **Constructing a shared subspace** via SVD of the task vectors, capturing the common direction of updates.
3. **Optimizing a learnable delta** $\delta$ projected onto the orthogonal complement of this subspace, allowing the task vectors to be adjusted without deviating from the intended update direction.
4. **Applying magnitude-based masking** (top-K) to keep only the most significant parameters.
5. **Combining the adjusted task vectors** with learned per-layer scaling factors.

### Subspace Construction

For each layer, the task vectors from all $N$ models are decomposed via SVD. Each task vector's SVD contributes a fraction ($1/N$) of its top singular vectors. These are combined and re-decomposed to obtain a shared subspace:

1. For each model $i$, compute SVD of its layer task vector: $\tau_i = U_i S_i V_i^T$.
2. Take the top $1/N$ singular vectors from each model and concatenate them into combined matrices $\bar{U}$, $\bar{S}$, $\bar{V}$.
3. Compute SVD of the combined left singular vectors: $\bar{U} = U_{\text{combined}} S_{\text{combined}} V_{\text{combined}}^T$.
4. The projection matrix uses the top `subspace` singular vectors: $P = U_{\text{combined}}[:, :k] U_{\text{combined}}[:, :k]^T$.

### Delta Optimization

The delta $\delta$ is initialized as zero and optimized using Adam. The gradient is projected onto the orthogonal complement of the shared subspace to ensure the delta only adjusts task vectors in directions that do not interfere with the primary update:

$$\nabla \delta \leftarrow \nabla \delta - P \nabla \delta = (I - P) \nabla \delta$$

This projection constraint ensures the delta operates in a complementary subspace, preserving the original task vector direction while resolving conflicts.

## Mathematical Formulation

### Task Vector Computation

For model $i$ and layer $l$, the task vector is:

$$\tau_{i,l} = \theta_{i,l} - \theta_{0,l}$$

Only layers matching specific filters are used (e.g., layers containing "encoder" and "weight" but excluding "layer_norm" for CLIP models).

### Loss Function

The optimization objective minimizes the pairwise conflict between task vectors (adjusted by delta and scaled by per-layer lambdas):

$$\mathcal{L} = \sum_j \sum_{k} \left( \lambda_j (\tau_{j} + \delta) \cdot \sum_{i} \lambda_i (\tau_i + \delta) + \tau_j^2 \right)^2$$

where $\lambda_i$ are per-layer, per-task scaling factors computed as:

$$\lambda_i^{(l)} = \frac{\lambda}{\|\tau_i^{(l)}\|}$$

### Gradient Projection

The gradient of the delta is projected:

$$g_\delta \leftarrow g_\delta - P^{(l)} g_\delta$$

where $P^{(l)}$ is the layer-specific projection matrix.

### Top-K Masking

After delta optimization, a top-K mask is applied to keep only the most significant parameters:

$$M = \mathbb{1}_{\{|\tau + \delta| \geq \text{kth\_value}\}}$$

where K represents the percentage of top parameters to retain (e.g., K=30 keeps the top 30% by magnitude).

### Final Merging

The final merged model is:

$$\theta = \theta_0 + \sum_i \lambda_i (\tau_i + \delta) \odot M$$

where $\odot$ denotes element-wise multiplication with the mask.

## Configuration

```yaml title="config/method/doge_ta/doge_ta.yaml"
--8<-- "config/method/doge_ta/doge_ta.yaml"
```

Key configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `subspace` | Number of singular vectors for the shared subspace | `6` |
| `K` | Top-K percentage for parameter masking (e.g., 30 for top 30%) | `30` |
| `lamda` | Scaling factor for lambda computation | `0.07` |

## Examples

### CLI Usage

Run DogeTA on CLIP models:

```bash
fusion_bench \
    method=doge_ta/doge_ta \
    method.subspace=6 \
    method.K=30 \
    method.lamda=0.07 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method.doge_ta.doge_ta import DOGE_TA_Algorithm

algorithm = DOGE_TA_Algorithm(subspace=6, K=30, lamda=0.07)

# modelpool must contain '_pretrained_' and fine-tuned models
merged_model = algorithm.run(modelpool)
```

## Related: Layer-Wise AdaMerging with DogeTA

The `layer_wise_adamerging.py` file provides an alternative DogeTA-based approach using the AdaMerging framework. The `LayerWiseAdaMergingAlgorithm` class constructs a `LayerWiseMergedModel` and performs test-time adaptation to learn layer-wise merging weights via entropy minimization. When used with DogeTA configs, it provides a data-driven alternative to the projection-based DogeTA approach.

### Configuration

The layer-wise AdaMerging variant can be invoked via:

```bash
fusion_bench \
    method=adamerging/clip \
    method.name=clip_layer_wise_adamerging_doge_ta \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

## Implementation Details

- **State dict filtering**: Only parameters matching `"encoder" in key and "layer_norm" not in key and "weight" in key` are used for task vector computation.
- **Delta initialization**: Delta is initialized as zeros matching the shape of task vectors.
- **Optimization**: Adam optimizer with learning rate 1e-4, 400 steps per layer.
- **Memory management**: The implementation explicitly tracks and reports memory usage, freeing the projection matrices and moving tensors to CPU after optimization.
- **Vector-state dict conversion**: Helper methods `state_dict_to_vector` and `vector_to_state_dict` handle the flattening and unflattening of parameters for top-K masking.

## References

[^1]: (arXiv 2025) Modeling Multi-Task Model Merging as Adaptive Projective Gradient Descent. http://arxiv.org/abs/2501.01230
