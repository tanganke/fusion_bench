# Tall Mask for Model Merging

Tall Mask is a parameter-efficient retrieval-based approach to model merging. Instead of producing a single merged model, Tall Mask generates task-specific binary masks that selectively retrieve parameters from a merged model. The key insight is that not all parameters in a merged model are useful for every task -- some parameters encode task-specific knowledge that conflicts with others. By applying a task-specific mask, each task can retrieve only the parameters that benefit it.

**The Mask Generation Rule**. Given a pretrained model with parameters $\theta_0$, a merged multi-task model with parameters $\theta_{mt}$, and a task-specific fine-tuned model with parameters $\theta_t$, the Tall Mask for task $t$ is defined as:

$$\text{mask}_t = \mathbb{I}\left( \|\theta_0 - \theta_t\|_1 > \lambda \cdot \|\theta_{mt} - \theta_t\|_1 \right)$$

where $\lambda$ (the `tall_mask_lambda` hyperparameter) controls the selectivity of the mask. Intuitively, this rule says: "keep a parameter for task $t$ if the pretrained model is farther from the task-specific model than the merged model is (up to a scaling factor $\lambda$)."

When $\lambda = 1.0$, the mask is strict -- only parameters where the merged model is genuinely closer to the task-specific model are retained. When $\lambda < 1.0$, the mask is more permissive. When $\lambda > 1.0$, the mask is more restrictive.

**Model Retrieval**. Once the Tall Mask is computed, the retrieved model for task $t$ is:

$$\theta_t^{\text{retrieved}} = \theta_0 + \text{mask}_t \odot (\theta_{mt} - \theta_0)$$

This effectively starts from the pretrained model and overlays only the task-relevant portions of the merged update vector.

## Multi-Task Vector Computation

The multi-task vector is computed as the sum of individual task vectors:

$$\tau_{\text{multi}} = \sum_{i} \tau_i = \sum_{i} (\theta_i - \theta_0)$$

The merged model parameters are then:

$$\theta_{mt} = \theta_0 + \tau_{\text{multi}}$$

This is equivalent to the Task Arithmetic merge (with scaling factor $\lambda = 1$).

## Examples

### CLI Usage

```yaml title="config/method/tall_mask/task_arithmetic.yaml"
--8<-- "config/method/tall_mask/task_arithmetic.yaml"
```

```bash
fusion_bench \
  method=tall_mask/task_arithmetic \
  method.tall_mask_lambda=0.6 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tall_mask_lambda` | float | 0.6 | Threshold scaling factor for mask generation. Lower values produce more permissive masks. |
| `debug` | int | 0 | Debug level. |
| `verbose` | int | 0 | Verbosity level. |

### API Usage

```python
from fusion_bench.method.tall_mask import TallMaskTaskArithmeticAlgorithm

algorithm = TallMaskTaskArithmeticAlgorithm(tall_mask_lambda=0.6)
result = algorithm.run(modelpool)
# result["models"] is a dict mapping task name -> retrieved model
```

## Output

The algorithm returns a dictionary with two keys:

- `"models"`: A dict mapping each task name to its retrieved model (pretrained model with the Tall Mask applied).
- `"metadata"`: Additional metadata (currently None).

Each retrieved model is a deep copy of the pretrained model with selectively applied merged parameters.

## Utility Functions

The `utils.py` module provides additional helper functions for advanced use cases:

- **`construct_tall_mask`**: Generates Tall Masks for multiple lambda values (0.2 through 0.6).
- **`find_optimal_mask`**: Selects the best lambda per task based on validation accuracy.
- **`construct_consensus_mask`**: Builds a consensus mask by counting activation frequency across tasks, enabling parameter pruning.

## Implementation Details

- [fusion_bench.method.tall_mask.task_arithmetic.TallMaskTaskArithmeticAlgorithm][]
- [fusion_bench.method.tall_mask.task_arithmetic.generate_task_masks][]
- [fusion_bench.method.tall_mask.utils.generate_task_masks][]

[^1]: Original implementation from: https://github.com/nik-dim/tall_masks/
[^2]: Adopted into FusionBench from: https://github.com/Zhou-Hangyu/randes/
