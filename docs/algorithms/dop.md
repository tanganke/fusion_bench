# DoP (Dual Projections)

DoP (Dual Projections for Balancing Stability and Plasticity) is a continual model merging algorithm that operates without access to training data. It sequentially merges fine-tuned models into a single model, using Singular Value Decomposition (SVD) to define projection subspaces for each task vector. By projecting weight differences onto these subspaces, DoP balances **stability** (preserving knowledge from previously merged models) and **plasticity** (incorporating knowledge from new models).

## Algorithm Overview

### Continual Merging Setting

DoP addresses the continual model merging scenario where models arrive sequentially and must be merged without storing all models in memory simultaneously. The algorithm processes models one at a time:

1. The first model is used as the initial merged model.
2. Each subsequent model is merged with the current merged result using layer-wise optimization.
3. The output after processing all models is the final merged model.

### SVD-Based Projection

For each fine-tuned model, DoP computes the task vector $\tau = \theta_{\text{fine-tuned}} - \theta_{\text{pretrained}}$ and performs SVD:

$$\tau = U S V^T$$

Only the top singular components (those capturing `svd_epsilon` fraction of the total energy) are retained, defining a low-rank projection subspace. This captures the "direction" in parameter space where the model was fine-tuned.

### Dual Projection Loss

For each layer, DoP defines a loss function that measures the distance between the merged weight and each fine-tuned weight in the SVD subspace. The "dual" refers to projecting onto both the left singular vectors (U) and right singular vectors (V):

$$\mathcal{L}_i^{(U)} = \| S^{1/2} U_{\text{main}}^T (\theta_{\text{merged}} - \theta_i) \|_F^2$$

$$\mathcal{L}_i^{(V)} = \| (\theta_{\text{merged}} - \theta_i) V_{\text{main}} S^{1/2} \|_F^2$$

The combined loss is:

$$\mathcal{L}_i = \mathcal{L}_i^{(U)} + \mathcal{L}_i^{(V)}$$

### Loss Balancing

Two strategies are available for balancing the losses from the merged model (stability) and the new model (plasticity):

1. **MGDA (Multi-Gradient Descent Algorithm)**: Uses Frank-Wolfe iteration to find optimal loss weights that minimize the norm of the combined gradient. With EMA (exponential moving average), the weights are smoothed over time.
2. **Fixed alpha**: Uses a fixed weight $\alpha$ for the merged model and $(1-\alpha)$ for the new model.

## Mathematical Formulation

### SVD Rank Selection

Given singular values $s_1 \geq s_2 \geq ... \geq s_r$, the rank $k$ is selected such that:

$$\frac{\sum_{j=1}^{k} s_j}{\sum_{j=1}^{r} s_j} \geq \epsilon$$

where $\epsilon$ is `svd_epsilon` (default 0.99999). Only the top $k$ singular triplets $(u_j, s_j, v_j)$ are kept.

### Projection Loss Computation

```python
def cal_loss_i(delta_tv, proj_s, proj_u, proj_v):
    proj_delta_1 = diag(S) @ U^T @ delta    # Left projection
    proj_delta_2 = delta @ V @ diag(S)       # Right projection
    loss_U = ||proj_delta_1||_F^2
    loss_V = ||proj_delta_2||_F^2
    return loss_U + loss_V    # or just one of them
```

### MGDA Optimization

When MGDA is enabled, the algorithm:

1. Computes gradients $\nabla \mathcal{L}_0$ and $\nabla \mathcal{L}_1$ for the merged and new model losses.
2. Normalizes gradients by loss values.
3. Finds the minimum-norm element in the convex hull of gradients (Frank-Wolfe).
4. Applies EMA smoothing: $\alpha_t = \beta \alpha_{t-1} + (1-\beta) \alpha_{\text{FW}}$.
5. Computes the combined loss: $\mathcal{L} = \alpha_t \mathcal{L}_0 + (1-\alpha_t) \mathcal{L}_1$.

### Projection Space Options

The `svd_proj_space` parameter controls which projection is used:

- **`uv`** (default): Both left and right projections.
- **`u`**: Only left projection (row space).
- **`v`**: Only right projection (column space).

## Configuration

### CLIP-Specific DOP

```yaml title="config/method/dop/dop.yaml"
--8<-- "config/method/dop/dop.yaml"
```

### General DOP (Architecture-Agnostic)

```yaml title="config/method/dop/dop_general.yaml"
--8<-- "config/method/dop/dop_general.yaml"
```

Key configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Learning rate for weight optimization | `1e-4` |
| `num_steps` | Optimization steps per layer | `200` |
| `mgda` | Use MGDA for loss balancing | `true` |
| `ema` | Apply EMA to MGDA weights | `true` |
| `ema_beta` | EMA decay rate | `0.999` |
| `alpha` | Fixed weight (if mgda=false) or initial weight (if mgda+ema) | `0.8` |
| `svd_epsilon` | Fraction of energy to retain in SVD | `0.99999` |
| `svd_proj_space` | Projection space: `u`, `v`, or `uv` | `uv` |
| `shuffle_order` | Shuffle model order before merging | `true` |
| `num_ray_actors` | Parallel Ray actors (0 = disabled) | `0` |

## Examples

### CLI Usage

Run DOP on CLIP models:

```bash
fusion_bench \
    method=dop/dop \
    method.shuffle_order=true \
    method.mgda=true \
    method.svd_epsilon=0.99999 \
    method.svd_proj_space=uv \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Run the general DOP with fixed alpha (no MGDA):

```bash
fusion_bench \
    method=dop/dop_general \
    method.mgda=false \
    method.alpha=0.7 \
    method.num_steps=300 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method.dop.dop_general import DOPMerging
from fusion_bench.modelpool import BaseModelPool

algorithm = DOPMerging(
    lr=1e-4,
    num_steps=200,
    mgda=True,
    ema=True,
    svd_epsilon=0.99999,
    svd_proj_space="uv",
)

merged_model = algorithm.run(modelpool)
```

## Implementation Details

### Two Implementations

1. **`ContinualDOPForCLIP`** (`dop.py`): CLIP-specific implementation with explicit support for CLIPVisionModel and CLIPVisionModelTaskPool. Includes options for saving intermediate models and step-wise evaluation.

2. **`DOPMerging`** (`dop_general.py`): Architecture-agnostic implementation that works with any model pool. Includes support for Ray-based parallel processing via `num_ray_actors`, dtype handling for SVD compatibility, and profiling.

### Layer-wise Optimization

Both implementations iterate through all leaf modules in the pretrained model. For `nn.Linear` layers with trainable weights (not in `exclude_keys`), they perform gradient-based optimization. For other layers (e.g., layer norms, embeddings), simple averaging is applied. Biases are always averaged.

### Memory Management

The algorithm processes one model at a time and deletes the fine-tuned model after merging (`del finetuned_model`), reducing memory pressure during continual merging.

### Ray Parallelism

The general implementation supports distributing layer-wise optimization across Ray actors. Each actor independently optimizes a linear layer's weights, enabling parallel processing of multiple layers.

## References

[^1]: (NeurIPS 2025) Continual Model Merging without Data: Dual Projections for Balancing Stability and Plasticity. http://arxiv.org/abs/2406.12345
