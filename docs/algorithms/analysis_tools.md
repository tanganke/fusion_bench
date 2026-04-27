# Analysis Tools

FusionBench provides a suite of analysis tools for examining task vectors -- the parameter differences between fine-tuned models and their pretrained base. These tools help researchers understand the geometric relationships between tasks, the distribution of parameter changes, and the compatibility of different task-specific updates.

## Task Vector Cosine Similarity

The `TaskVectorCosSimilarity` algorithm computes pairwise cosine similarity between the task vectors of all models in a model pool. This reveals how aligned or orthogonal different tasks are in parameter space.

**Computation**. For each fine-tuned model $i$, the task vector is:

$$\tau_i = \text{flatten}(\theta_i - \theta_0)$$

where $\theta_i$ are the fine-tuned parameters and $\theta_0$ are the pretrained parameters. The pairwise cosine similarity between tasks $i$ and $j$ is:

$$\text{cos\_sim}(i, j) = \frac{\tau_i \cdot \tau_j}{\|\tau_i\| \|\tau_j\|}$$

**Interpretation**:
- **High similarity** (near 1.0): Tasks update parameters in similar directions; merging these tasks is likely to be effective.
- **Low similarity** (near 0.0 or negative): Tasks update parameters in different directions; merging may cause interference.

### CLI Usage

```yaml title="config/method/analysis/task_vector_cos_similarity.yaml"
--8<-- "config/method/analysis/task_vector_cos_similarity.yaml"
```

```bash
fusion_bench \
  method=analysis/task_vector_cos_similarity \
  method.plot_heatmap=true \
  method.trainable_only=true \
  method.max_points_per_model=null \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=dummy
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot_heatmap` | bool | true | Generate a heatmap visualization (saved as PDF). |
| `trainable_only` | bool | true | Only use trainable parameters. |
| `max_points_per_model` | int | null | Max parameters to sample per model for memory efficiency. |
| `output_path` | str | null | Output directory. Defaults to fabric logger log_dir. |

### Output

- `task_vector_cos_similarity.csv`: Pairwise cosine similarity matrix.
- `task_vector_cos_similarity.pdf`: Heatmap visualization (if `plot_heatmap=true`).

### API Usage

```python
from fusion_bench.method.analysis import TaskVectorCosSimilarity

algorithm = TaskVectorCosSimilarity(
    plot_heatmap=True,
    trainable_only=True,
    max_points_per_model=None,
    output_path="./analysis",
)
pretrained_model = algorithm.run(modelpool)
```

## Task Vector Violin Plot

The `TaskVectorViolinPlot` algorithm creates violin plots visualizing the distribution of task vector values across all models in the pool. This reveals how each task's parameter changes are distributed -- whether they are concentrated near zero or spread widely.

**Two Visualizations**:
1. **Raw values**: Shows the full distribution including positive and negative changes, revealing the direction of parameter updates.
2. **Absolute values**: Shows the magnitude of changes regardless of direction, useful for comparing the overall scale of updates across tasks.

### CLI Usage

```yaml title="config/method/analysis/task_vector_violin_plot.yaml"
--8<-- "config/method/analysis/task_vector_violin_plot.yaml"
```

```bash
fusion_bench \
  method=analysis/task_vector_violin_plot \
  method.trainable_only=true \
  method.max_points_per_model=1000 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=dummy
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trainable_only` | bool | true | Only use trainable parameters. |
| `max_points_per_model` | int | 1000 | Max parameters to sample per model. |
| `fig_kwargs` | dict | null | Matplotlib subplots kwargs (e.g., `{'figsize': (12, 8)}`). |
| `output_path` | str | null | Output directory. Defaults to fabric logger log_dir. |

### Output

- `task_vector_violin.pdf`: Violin plot of raw task vector distributions.
- `task_vector_violin_abs.pdf`: Violin plot of absolute task vector values.

### API Usage

```python
from fusion_bench.method.analysis import TaskVectorViolinPlot

algorithm = TaskVectorViolinPlot(
    trainable_only=True,
    max_points_per_model=5000,
    fig_kwargs={'figsize': (12, 8), 'dpi': 300},
    output_path="./analysis_plots",
)
pretrained_model = algorithm.run(modelpool)
```

## Downsampling for Large Models

For models with billions of parameters, computing task vectors over all parameters can be memory-intensive. Both tools support `max_points_per_model` to randomly subsample a fixed number of parameter values, preserving the statistical distribution while reducing memory usage.

## Implementation Details

- [fusion_bench.method.analysis.task_vector_cos_similarity.TaskVectorCosSimilarity][]
- [fusion_bench.method.analysis.task_vector_violin_plot.TaskVectorViolinPlot][]

[^1]: (2024) Efficient and Effective Weight-Ensembling Mixture of Experts for Multi-Task Model Merging. http://arxiv.org/abs/2410.21804
