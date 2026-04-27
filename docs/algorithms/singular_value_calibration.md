# Singular Value Calibration

When merging multiple fine-tuned models, a fundamental challenge arises: parameters that encode shared knowledge across tasks get "over-counted" -- they are summed multiple times in the merged model, leading to spectral over-accumulation that degrades performance. Singular Value Calibration (SVC) addresses this by analyzing the merged model in the spectral domain and selectively scaling down subspaces where over-accumulation occurs.

**The Core Problem**. Consider merging $K$ task-specific models via simple averaging or task arithmetic. For weight matrices, the merged weight can be expressed as:

$$W_{\text{merge}} = W_{\text{pre}} + \frac{1}{K}\sum_{i=1}^{K} \Delta W_i$$

where $W_{\text{pre}}$ is the pretrained base weight and $\Delta W_i = W_i - W_{\text{pre}}$ is the task-specific update. When multiple tasks update the same spectral subspace (e.g., the same singular direction), the merged response along that subspace becomes amplified, potentially distorting the shared representation.

**Spectral Analysis via SVD**. SVC decomposes the merged update $\Delta W_{\text{merge}} = W_{\text{merge}} - W_{\text{pre}}$ using Singular Value Decomposition:

$$\Delta W_{\text{merge}} = U \Sigma V^H$$

Each left singular vector $u_r$ in $U$ defines a spectral subspace. The projection of each task's update onto this subspace yields a "subspace response":

$$a_r^i = u_r^T \Delta W_i$$

**Interference Measurement**. For each subspace $r$ and task $i$, SVC computes a projection coefficient that quantifies how the merged response scales along the task's direction:

$$s_r^i = \frac{\langle a_r^{\text{merge}}, a_r^i \rangle}{\|a_r^i\|^2}$$

When $s_r^i > 1$, the merged response amplifies that subspace relative to the task's original response, indicating over-counting. When $s_r^i < 1$, the response is attenuated.

**Calibration**. SVC defines a per-subspace calibration factor:

$$\gamma_r = \frac{K}{\sum_{i=1}^{K} \max(\alpha, s_r^i)}$$

where $\alpha$ is a user-tunable hyperparameter that controls calibration strength. Higher $\alpha$ values lead to more aggressive calibration (greater scaling down), while lower values retain more of the original merged responses. The calibrated singular values are then:

$$S_{\text{calibrated}} = \gamma \odot \Sigma$$

and the calibrated merged weight is reconstructed as:

$$W_{\text{calibrated}} = U \text{diag}(S_{\text{calibrated}}) V^H + W_{\text{pre}}$$

## Implementation Details

FusionBench provides three variants of SVC:

1. **`SingularValueCalibration`** (`svc.py`): The base calibration algorithm. It requires a pre-merged model (with key `_merged_`) in the model pool. It iterates over all 2D weight matrices, performs SVD-based calibration, and updates the merged model in place. Both an original loop-based and an accelerated vectorized implementation are available (controlled via the `FUSION_BENCH_SVC_IMPL` environment variable).

2. **`SingularValueCalibrationWithBaseMethod`** (`generic.py`): A generic wrapper that first runs an arbitrary base merging method (e.g., Simple Average, TIES-Merging, DARE) and then applies SVC calibration to the result. The base method is specified via the `base_method` config field.

3. **`SingularValueCalibrationArithmeticTask`** (`task_arithmetic.py`): A convenience variant that combines Task Arithmetic with a configurable `scaling_factor` followed by SVC calibration.

All variants operate layer-by-layer on 2D weight matrices only (biases and LayerNorm weights are left unchanged) and support GPU acceleration via the `accelerator` parameter.

## Examples

### CLI Usage -- SVC after Task Arithmetic

```yaml title="config/method/singular_value_calibration/task_arithmetic.yaml"
--8<-- "config/method/singular_value_calibration/task_arithmetic.yaml"
```

```bash
fusion_bench \
  method=singular_value_calibration/task_arithmetic \
  method.scaling_factor=0.3 \
  method.alpha=0.1 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### CLI Usage -- SVC with Generic Base Method

```yaml title="config/method/singular_value_calibration/simple_average.yaml"
--8<-- "config/method/singular_value_calibration/simple_average.yaml"
```

```bash
fusion_bench \
  method=singular_value_calibration/simple_average \
  method.alpha=0.1 \
  method.base_method._target_=fusion_bench.method.SimpleAverageAlgorithm \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
import torch
from fusion_bench.method.singular_value_calibration import SingularValueCalibration

algorithm = SingularValueCalibration(alpha=0.1)

# modelpool must contain '_merged_' (pre-merged model) and '_pretrained_'
calibrated_model = algorithm.run(modelpool)
```

### Accelerated Implementation

By default, the vectorized accelerated implementation is used. To use the original loop-based implementation:

```bash
FUSION_BENCH_SVC_IMPL=original fusion_bench method=singular_value_calibration/task_arithmetic ...
```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.1 | Calibration strength. Higher values lead to more aggressive spectral scaling down. |
| `accelerator` | str/None | None | Device for computation (e.g., `"cuda"`). Auto-detects if None. |
| `scaling_factor` | float | 0.3 | (Arithmetic variant only) Scaling factor for task arithmetic before calibration. |
| `base_method` | Dict | SimpleAverage | (Generic variant only) Config dict for the base merging algorithm. |

## Implementation Details

- [fusion_bench.method.singular_value_calibration.svc.SingularValueCalibration][]
- [fusion_bench.method.singular_value_calibration.generic.SingularValueCalibrationWithBaseMethod][]
- [fusion_bench.method.singular_value_calibration.task_arithmetic.SingularValueCalibrationArithmeticTask][]
- [fusion_bench.method.singular_value_calibration.utils.subspace_consistency_spectral_calibration][]
- [fusion_bench.method.singular_value_calibration.utils.subspace_consistency_spectral_calibration_accelerated][]

[^1]: (2026) When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging. http://arxiv.org/abs/2602.05536
