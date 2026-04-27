# OPCM (Orthogonal Projection for Continual Merging)

OPCM addresses the continual model merging problem, where models arrive sequentially and must be merged one by one into a single model. The core insight is that when merging a new task model into an existing merged model, the task vectors often have overlapping components that cause negative interference. OPCM uses Singular Value Decomposition (SVD) to project each new task vector into the subspace orthogonal to the dominant directions of the current merged task vector[^1].

<figure markdown="span">
  ![OPCM](images/opcm.png){ width="600" }
  <figcaption>OPCM: SVD-based orthogonal projection for continual merging. Credit to <sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></figcaption>
</figure>

**The Continual Merging Setting**. In the continual (or online) setting, models $\theta_1, \theta_2, \dots, \theta_K$ arrive sequentially. After merging the first $t$ models, the merged model $\theta^{(t)}$ must be updated to incorporate $\theta_{t+1}$ without reprocessing $\theta_1, \dots, \theta_t$. OPCM maintains a running merge that preserves previous task knowledge while incorporating new tasks.

## Mathematical Formulation

Let $\theta_0$ be the pretrained (anchor) model. Given the current merged model $\theta^{(t-1)}$ and a new task model $\theta_t$, OPCM computes:

1. **Task Vectors**:
   $$\tau_{merged}^{(t-1)} = \theta^{(t-1)} - \theta_0$$
   $$\tau_{new} = \theta_t - \theta_0$$

2. **SVD of Merged Task Vector** (for linear weight matrices):
   $$\tau_{merged}^{(t-1)} = U \Sigma V^\top$$

3. **Projection and Zeroing**:
   Project the new task vector into the SVD basis of the merged task vector:
   $$P = U^\top \tau_{new} V$$
   Zero out the diagonal of $P$ to remove alignment with singular vectors:
   $$P_{ii} \leftarrow 0$$
   Zero out the top-left block (dominant singular directions) up to a split rank $r$:
   $$P_{ij} \leftarrow 0 \quad \text{for } i, j \leq r$$

   The split rank $r$ is determined by the parameter $\alpha$: it is the smallest index such that the cumulative sum of singular values exceeds a fraction $\alpha$ of the total:
   $$r = \min \left\{ k : \frac{\sum_{i=1}^k \sigma_i}{\sum_{i} \sigma_i} > \alpha \right\}$$

4. **Reconstruct the Cleaned Task Vector**:
   $$\tilde{\tau}_{new} = U P V^\top$$

5. **Update the Merged Model**:
   $$\theta^{(t)} = \theta_0 + \frac{\lambda_{t-1} \tau_{merged}^{(t-1)} + \tilde{\tau}_{new}}{\lambda_t}$$

   where $\lambda_t$ is a scaling factor computed to maintain a stable task vector norm, approximately growing as $\sqrt{t}$.

For non-linear parameters (biases, LayerNorm weights), OPCM uses a simpler averaging formula without SVD projection:
$$\theta^{(t)} = \theta_0 + \frac{\lambda_{t-1} (\theta^{(t-1)} - \theta_0) + (\theta_t - \theta_0)}{\lambda_t}$$

## OPCM Variants in FusionBench

### 1. OPCM (General)

The `OPCM` class in `opcm_general.py` is the general-purpose implementation with support for Ray-based distributed merging.

```python
OPCM(
    alpha=0.5,                    # SVD projection threshold
    shuffle_order=True,           # shuffle model order
    seed=None,                    # random seed
    save_on_every_step=True,      # save checkpoint at each step
    evaluate_on_every_step=False, # evaluate at each step
    num_ray_actors=0,             # parallel actors for distributed merge
)
```

### 2. OPCM for CLIP

The `OPCMForCLIP` class in `opcm.py` is a CLIP-specific implementation that includes task pool integration for evaluation at each merging step.

```python
OPCMForCLIP(
    alpha=0.5,
    shuffle_order=True,
    seed=None,
    save_on_every_step=True,
    evaluate_on_every_step=True,
)
```

### 3. Continual Task Arithmetic

A baseline continual merging method that applies task arithmetic incrementally without SVD projection.

```python
ContinualTaskArithmeticForCLIP(
    scaling_factor=0.3,
    shuffle_order=True,
    seed=None,
    save_on_every_step=True,
    evaluate_on_every_step=True,
)
```

### 4. Continual TIES-Merging

A continual variant of TIES-Merging that resolves sign conflicts at each step.

```python
ContinualTiesMergingForCLIP(
    scaling_factor=0.5,
    threshold=20,
    remove_keys=[],
    merge_func="sum",
    shuffle_order=True,
    seed=None,
    save_on_every_step=True,
    evaluate_on_every_step=True,
)
```

### 5. Continual Weight Average

A simple incremental averaging baseline: $\theta^{(t)} = \frac{t \cdot \theta^{(t-1)} + \theta_t}{t + 1}$.

```python
ContinualWeightAverageForCLIP(
    shuffle_order=True,
    seed=None,
    save_on_every_step=True,
    evaluate_on_every_step=True,
)
```

## SVD Projection Details

The SVD projection is the core of OPCM. The key idea is:

1. **Dominant directions** of the merged task vector represent the "shared" knowledge accumulated so far.
2. **New task vectors** projected onto these dominant directions cause interference.
3. By **zeroing out** the projection onto the top-$r$ singular directions, we keep the new information that is *orthogonal* (novel) to what has been merged.
4. The **diagonal zeroing** step removes alignment with each singular vector individually, preventing self-reinforcement.

The parameter $\alpha$ controls the aggressiveness:
- $\alpha = 0.5$ means we zero out directions explaining 50% of the variance.
- Higher $\alpha$ zeros out more directions, being more conservative about what gets merged.
- Lower $\alpha$ zeros out fewer directions, allowing more overlap.

## Examples

### CLI Usage

Configuration template for OPCM:

```yaml title="config/method/opcm/opcm.yaml"
--8<-- "config/method/opcm/opcm.yaml"
```

Run OPCM for CLIP:

```bash
fusion_bench method=opcm/opcm \
    method.alpha=0.5 \
    method.shuffle_order=true \
    method.evaluate_on_every_step=true \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Configuration for the general OPCM (with Ray support):

```yaml title="config/method/opcm/opcm_general.yaml"
--8<-- "config/method/opcm/opcm_general.yaml"
```

Run general OPCM with distributed merging:

```bash
fusion_bench method=opcm/opcm_general \
    method.alpha=0.5 \
    method.num_ray_actors=4 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Configuration for continual task arithmetic:

```yaml title="config/method/opcm/task_arithmetic.yaml"
--8<-- "config/method/opcm/task_arithmetic.yaml"
```

Run continual task arithmetic:

```bash
fusion_bench method=opcm/task_arithmetic \
    method.scaling_factor=0.3 \
    method.shuffle_order=true \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Configuration for continual TIES-Merging:

```yaml title="config/method/opcm/ties_merging.yaml"
--8<-- "config/method/opcm/ties_merging.yaml"
```

Run continual TIES-Merging:

```bash
fusion_bench method=opcm/ties_merging \
    method.scaling_factor=0.5 \
    method.threshold=20 \
    method.merge_func=sum \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Configuration for continual weight average:

```yaml title="config/method/opcm/weight_average.yaml"
--8<-- "config/method/opcm/weight_average.yaml"
```

Run continual weight average:

```bash
fusion_bench method=opcm/weight_average \
    method.shuffle_order=true \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method.opcm import OPCM

# Instantiate the algorithm
algorithm = OPCM(
    alpha=0.5,
    shuffle_order=True,
    save_on_every_step=True,
    evaluate_on_every_step=False,
)

# Run continual merging on a model pool
merged_model = algorithm.run(modelpool)
```

## Runtime Behavior

All OPCM variants share these behavioral features:

1. **Sequential Processing**: Models are processed one at a time. The order can be shuffled (`shuffle_order=True`) or fixed.
2. **Checkpoint Saving**: When `save_on_every_step=True`, intermediate merged models are saved to `{log_dir}/checkpoints/merged_model_{step}/`.
3. **Evaluation**: When `evaluate_on_every_step=True`, the merged model is evaluated after each step. Reports are saved as `report_{step}.json`.
4. **Logging**: TensorBoard logs track task vector norms, $\lambda_t$ values, and other metrics.
5. **Progress Bars**: Uses `tqdm` for both model-level and layer-level progress tracking.
6. **Profiling**: Built-in `SimpleProfilerMixin` reports timing for loading, merging, saving, and evaluation.

## Distributed Merging with Ray

The general `OPCM` class supports distributed merging via Ray actors. Set `num_ray_actors > 0` to spawn Ray actors that process layers in parallel:

```bash
fusion_bench method=opcm/opcm_general \
    method.num_ray_actors=8 \
    ...
```

Each Ray actor independently computes the SVD projection for assigned linear layers, while the main process handles bias and non-linear parameters.

## Utility Functions

Key utilities in `fusion_bench/method/opcm/utils.py`:

- **`svd(w, full_matrices, accelerator)`**: Performs SVD on a weight tensor with optional device transfer.
- **`frobenius_inner_product(w1, w2)`**: Computes the Frobenius inner product of two matrices.
- **`get_task_vector_norm(model, pretrained_model)`**: Computes the L2 norm of the task vector.

## Hyperparameter Guidelines

- **`alpha`**: Controls the fraction of singular directions to zero out. Typical values: 0.3-0.7. Higher values are more conservative, zeroing out more of the new task's overlap with existing knowledge.
- **`shuffle_order`**: Randomizing model order can help prevent bias toward earlier models. Set `seed` for reproducibility.

## Implementation Details

- [OPCM][fusion_bench.method.opcm.OPCM]
- [OPCMForCLIP][fusion_bench.method.opcm.OPCMForCLIP]
- [ContinualTaskArithmeticForCLIP][fusion_bench.method.opcm.ContinualTaskArithmeticForCLIP]
- [ContinualTiesMergingForCLIP][fusion_bench.method.opcm.ContinualTiesMergingForCLIP]
- [ContinualWeightAverageForCLIP][fusion_bench.method.opcm.ContinualWeightAverageForCLIP]

[^1]: (ICML 2024) Tang et al. Merging Multi-Task Models via Weight-Ensembling Mixture of Experts. https://arxiv.org/abs/2402.00433
