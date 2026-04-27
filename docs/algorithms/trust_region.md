# Trust Region for Model Merging

Trust Region is a training-free model merging approach that identifies and navigates knowledge conflicts between task vectors. The core idea is to construct a "trust region" mask that identifies parameter dimensions where tasks conflict with each other, then zero out the conflicting portions of each task vector before merging.

**The Conflict Metric**. For each pair of tasks $(i, j)$, the algorithm computes a conflict measure:

$$\Omega = \sum_{i \neq j} |\bar{g}_i| \odot |\tau_j|$$

where $\bar{g}_i$ is the average absolute gradient for task $i$ (computed on its training data), $\tau_j$ is the task vector for task $j$, and $\odot$ denotes element-wise multiplication. The matrix $\Omega$ is a flattened vector where each element represents the cumulative conflict across all task pairs at that parameter dimension.

**Trust Region Mask**. A threshold is applied to $\Omega$ to create a binary mask:

$$\text{mask} = \mathbb{I}(\Omega < \text{threshold})$$

The threshold is set to the $q$-th quantile of $\Omega$ values (controlled by `threshold_quantile`). Parameters with conflict values below the threshold are considered "safe" and are preserved; parameters above the threshold are deemed conflicting and are zeroed out.

**Task Vector Masking**. Each task vector is masked:

$$\tau_i^{\text{masked}} = \tau_i \odot \text{mask}$$

Then the masked task vectors are summed and added to the pretrained model:

$$\theta = \theta_0 + \lambda \sum_i \tau_i^{\text{masked}}$$

**Gradient Computation**. The average absolute gradient for each task is computed by:
1. Taking the pretrained model (initialized from scratch for each task)
2. Computing per-sample gradients on the task's training data (up to `max_samples` samples)
3. Averaging the absolute gradient values across all samples

In zero-shot mode (`zero_shot=true`), the task vector's absolute values are used as a proxy for gradients, eliminating the need for training data.

## Examples

### CLI Usage

```yaml title="config/method/trust_region/clip_task_arithmetic.yaml"
--8<-- "config/method/trust_region/clip_task_arithmetic.yaml"
```

```bash
fusion_bench \
  method=trust_region/clip_task_arithmetic \
  method.scaling_factor=0.3 \
  method.threshold_quantile=0.99 \
  method.max_samples=128 \
  method.batch_size=128 \
  method.zero_shot=false \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### Zero-Shot Mode

When training data is unavailable, use zero-shot mode which substitutes gradients with task vector magnitudes:

```bash
fusion_bench \
  method=trust_region/clip_task_arithmetic \
  method.zero_shot=true \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scaling_factor` | float or list[float] | 0.3 | Scaling factor for the merged task vector. Can be a single value (returns one model) or a list (returns a dict of models). |
| `threshold_quantile` | float | 0.99 | Quantile of the conflict metric used as the trust region threshold. Lower values are more aggressive. |
| `max_samples` | int | 128 | Maximum number of training samples per task for gradient computation. |
| `batch_size` | int | 128 | Batch size for gradient computation. |
| `zero_shot` | bool | false | If true, use task vector abs as gradient proxy instead of computing actual gradients. |

### API Usage

```python
from fusion_bench.method.trust_region import TaskArithmeticWithTrustRegionForCLIP

algorithm = TaskArithmeticWithTrustRegionForCLIP(
    scaling_factor=0.3,
    threshold_quantile=0.99,
    max_samples=128,
    batch_size=128,
    zero_shot=False,
)
merged_model = algorithm.run(modelpool)
```

## Output

- When `scaling_factor` is a single float: returns the merged model directly.
- When `scaling_factor` is a list of floats: returns a dict mapping each scaling factor to its merged model, enabling hyperparameter search.

## Implementation Details

- [fusion_bench.method.trust_region.clip_task_arithmetic.TaskArithmeticWithTrustRegionForCLIP][]
- [fusion_bench.method.trust_region.utils.state_dict_to_vector][]
- [fusion_bench.method.trust_region.utils.vector_to_state_dict][]

[^1]: (2024) Task Arithmetic in Trust Region: A Training-Free Model Merging Approach to Navigate Knowledge Conflicts. https://openreview.net/forum?id=q3ztjJRQuJ
