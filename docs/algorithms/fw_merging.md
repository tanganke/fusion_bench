# Fisher Whitelister (FW Merging)

Fisher Whitelister is an iterative model merging algorithm based on the Frank-Wolfe (conditional gradient) optimization framework. At each iteration, it selects the fine-tuned model whose parameters most strongly align with the current gradient of the loss, then merges it into the running model. This greedy selection strategy effectively builds a "whitelist" of the most useful models for the merging objective.

The algorithm comes in two variants: **Hard** and **Soft**, each with different merging strategies and optimization approaches.

## Algorithm Overview

### The Frank-Wolfe Framework

The Frank-Wolfe algorithm solves constrained optimization problems by iteratively:

1. Computing the gradient of the objective at the current solution.
2. Finding the extreme point (in this case, a model checkpoint) that most minimizes the linear approximation of the objective.
3. Moving toward that point with a decaying step size.

In model merging, the "extreme points" are the fine-tuned model parameters, and the "gradient" is computed from the loss on the target tasks.

### Model Selection Criterion

At each iteration, the algorithm computes the gradient of a loss function (cross-entropy or entropy) w.r.t. the merged model's parameters. It then selects the fine-tuned model whose state dictionary has the minimum inner product (maximum negative alignment) with the gradient:

$$\text{score}(m) = \sum_{p} \frac{\nabla_p \mathcal{L} \cdot \theta_{m,p}}{\|\nabla_p \mathcal{L}\| \cdot \|\theta_{m,p}\|}$$

The model minimizing this cosine similarity sum is selected, as it points most directly downhill in the loss landscape.

### Granularity

Two granularity levels are supported:

- **`task`** (default): Select one model per iteration for the entire model.
- **`layer`**: Select the best model per parameter/layer independently, constructing a composite model from different sources.

## Hard Variant (FrankWolfeHardAlgorithm)

### Merging Functions

The Hard variant supports two merging backbones:

1. **Task Arithmetic**: Merges using task vectors scaled by the Frank-Wolfe step size.
2. **TIES Merging**: Applies TIES (Trimmed, Eliminated, and Signed) merging with configurable threshold for conflict resolution.

### Iteration Process

1. Compute loss gradients over `dataset_size` samples from each task.
2. Select the model with minimum gradient alignment.
3. Determine step size: $\gamma_t = \frac{2}{t+2} \cdot \text{step\_size}$.
4. Merge the selected model into the current merged model using the chosen merge function.

### Initialization

The merged model can be initialized in three ways:

- **`base`**: Start from the pretrained model.
- **Empty string (`""`)**: Start from a merge of all models using the merge function.
- **File path**: Load layer-wise weights from a saved tensor file.

## Soft Variant (FrankWolfeSoftAlgorithm)

### Key Differences

The Soft variant extends the Hard variant with:

1. **AdaMerging integration**: Instead of a static merge, the Soft variant can use AdaMerging to learn optimal merging weights via test-time adaptation after each Frank-Wolfe iteration.
2. **Per-task model selection**: Models are selected independently for each task within an iteration, then all selected models are merged together.
3. **AdaMerging as merge function**: When `merge_fn="adamerging"`, the selected models are combined using layer-wise AdaMerging with entropy minimization.

### Iteration Process (Soft with AdaMerging)

1. For each task, compute gradients and select the best-aligned model.
2. Construct a `LayerWiseMergedModel` with all selected models.
3. Run AdaMerging for `ada_iters` steps to optimize layer-wise weights.
4. The result becomes the new merged model.

### Projection onto Simplex

The Soft variant includes a `projection_simplex_sort` utility that projects a vector onto the probability simplex, ensuring non-negative weights that sum to one. This is used in the AdaMerging weight optimization.

## Mathematical Formulation

### Gradient Computation

For each task $t$, the gradient is computed over `dataset_size` samples:

$$\nabla \mathcal{L} = \frac{1}{D \cdot T} \sum_{d=1}^{D} \nabla_\theta \ell(f_\theta(x_d), y_d)$$

where $D$ is `dataset_size` and $T$ is the number of tasks.

### Frank-Wolfe Step Size

The step size follows the standard Frank-Wolfe schedule:

$$\gamma_t = \frac{2}{t + 2} \cdot \alpha$$

where $\alpha$ is the `step_size` hyperparameter and $t$ is the iteration index.

### Task Arithmetic Merge

$$\theta_{t+1} = \theta_t + \gamma_t \cdot (\theta_{\text{selected}} - \theta_{\text{pretrained}})$$

### TIES Merge

TIES merging applies sign-based conflict resolution and magnitude thresholding before summing task vectors.

### AdaMerging (Soft variant)

For each layer $l$, learn weights $w^{(l)}$ by:

$$\min_{w^{(l)}} \mathcal{L}_{\text{entropy}}\left( \sum_j w^{(l)}_j \cdot (\theta_j^{(l)} - \theta_0^{(l)}) + \theta_0^{(l)} \right)$$

subject to $w^{(l)} \in [0, 1]$ and $\sum_j w^{(l)}_j = 1$ (if `tie_weights` is enabled).

## Configuration

### Hard Variant

```yaml title="config/method/fw_merging/fw_hard.yaml"
--8<-- "config/method/fw_merging/fw_hard.yaml"
```

### Soft Variant

```yaml title="config/method/fw_merging/fw_soft.yaml"
--8<-- "config/method/fw_merging/fw_soft.yaml"
```

Key configuration parameters:

| Parameter | Description | Hard Default | Soft Default |
|-----------|-------------|-------------|-------------|
| `merge_fn` | Merge function: `task_arithmetic`, `ties`, or `adamerging` | `task_arithmetic` | `adamerging` |
| `max_iters` | Number of Frank-Wolfe iterations | `10` | `10` |
| `step_size` | Frank-Wolfe step size multiplier | `0.1` | `0.1` |
| `dataset_size` | Samples per task for gradient computation | `100` | `100` |
| `granularity` | Selection granularity: `task` or `layer` | `task` | `task` |
| `init_weight` | Initialization: `base`, file path, or empty | `""` | `""` |
| `ada_iters` | AdaMerging iterations (Soft only) | N/A | `500` |
| `ada_coeff` | AdaMerging initial weight coefficient | N/A | `1e-8` |
| `ada_loss` | AdaMerging loss: `entropy_loss` or `cross_entropy` | N/A | `entropy_loss` |

## Examples

### CLI Usage

Hard variant with Task Arithmetic:

```bash
fusion_bench \
    method=fw_merging/fw_hard \
    method.merge_fn=task_arithmetic \
    method.max_iters=10 \
    method.step_size=0.1 \
    method.granularity=task \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Hard variant with TIES merging:

```bash
fusion_bench \
    method=fw_merging/fw_hard \
    method.merge_fn=ties \
    method.threshold=20 \
    method.scaling_factor=0.3 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Soft variant with AdaMerging:

```bash
fusion_bench \
    method=fw_merging/fw_soft \
    method.merge_fn=adamerging \
    method.ada_iters=500 \
    method.max_iters=10 \
    method.granularity=task \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method.fw_merging.fw_hard import FrankWolfeHardAlgorithm

algorithm = FrankWolfeHardAlgorithm(
    merge_fn="task_arithmetic",
    step_size=0.1,
    max_iters=10,
    dataset_size=100,
    granularity="task",
)

merged_model = algorithm.run(modelpool)
```

## Implementation Details

- **Gradient computation**: The `frank_wolfe_iteration` method computes gradients over a subset of training data. The loss is normalized by `dataset_size * number_of_tasks`.
- **Model selection**: The `frank_wolfe_selection` method computes cosine similarity between gradients and model parameters for each candidate model.
- **State preservation**: The `set_requires_grad` method in the Soft variant preserves the original pretrained model's gradient requirements on the merged model.
- **Loss functions**: Both variants support cross-entropy (using labels) and entropy (label-free) loss functions. The Soft variant's AdaMerging uses the `ada_loss` parameter to switch between them.

## References

[^1]: Frank-Wolfe algorithm for conditional gradient optimization. Provides the theoretical foundation for iterative model selection and merging.
[^2]: (ICLR 2023) Editing Models with Task Arithmetic. http://arxiv.org/abs/2212.04089
[^3]: (ICLR 2023) TIES-Merging: Resolving Interference in Model Parameter Upstream. http://arxiv.org/abs/2303.09922
