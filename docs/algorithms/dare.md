# DARE (Distributed Representations for model mergEd models)

In model merging, a key challenge is that task-specific updates from different fine-tuned models often occupy overlapping parameter regions, causing destructive interference when simply averaged or added. DARE addresses this by randomly dropping and rescaling elements of each model's task vector before merging, effectively distributing each model's contribution across disjoint parameter subsets[^1].

<figure markdown="span">
  ![DARE](images/dare.png){ width="500" }
  <figcaption>DARE: Random drop and rescale of task vectors before merging. Credit to <sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></figcaption>
</figure>

**Core Idea**. For each fine-tuned model, DARE constructs its task vector $\tau_i = \theta_i - \theta_0$ relative to the pretrained model $\theta_0$. It then randomly drops a fraction $p$ (sparsity ratio) of the elements in each task vector and rescales the remaining elements by $1/(1-p)$ to maintain the expected sum. Because the drops are independent across models, each parameter position predominantly carries the signal of a single model, reducing negative interference.

## Mathematical Formulation

Given a pretrained model $\theta_0$ and $K$ fine-tuned models $\{\theta_i\}_{i=1}^K$, DARE proceeds as follows:

1. **Task Vector Construction**: For each model $i$, compute the task vector:
   $$\tau_i = \theta_i - \theta_0$$

2. **Random Drop and Rescale**: For each element in $\tau_i$, apply a random binary mask $M_i \in \{0, 1\}^{|\theta|}$ where each element is independently set to 0 with probability $p$ (the sparsity ratio):
   $$\tilde{\tau}_i = \frac{M_i \odot \tau_i}{1 - p}$$
   where $\odot$ denotes element-wise multiplication, and the rescaling factor $1/(1-p)$ ensures $\mathbb{E}[\tilde{\tau}_i] = \tau_i$.

3. **Merging**: The merged model is obtained by summing (or applying a merge function to) the pruned task vectors and adding to the pretrained model:
   $$\theta = \theta_0 + \lambda \sum_{i=1}^{K} \tilde{\tau}_i$$
   where $\lambda$ is the scaling factor.

The implementation supports two modes:
- **Full-parameter DARE**: The random drop is applied to all parameters (`only_on_linear_weights=False`).
- **Linear-only DARE**: The random drop is applied only to the weight matrices of `nn.Linear` layers (`only_on_linear_weights=True`), which is the default setting in the original paper.

## Variants

FusionBench implements three DARE variants:

### 1. DARE + Simple Average

The simplest DARE variant applies random drop-and-rescale to each model independently, then averages the pruned task vectors with equal weights.

```python
DareSimpleAverage(
    sparsity_ratio=0.5,       # fraction of elements to drop
    only_on_linear_weights=False,
    rescale=True,             # rescale remaining elements by 1/(1-p)
)
```

### 2. DARE + Task Arithmetic

DARE combined with Task Arithmetic, where pruned task vectors are summed and scaled by a coefficient $\lambda$.

```python
DareTaskArithmetic(
    scaling_factor=0.3,       # the lambda coefficient
    sparsity_ratio=0.5,
    only_on_linear_weights=False,
    rescale=True,
)
```

### 3. DARE + TIES-Merging

DARE combined with TIES-Merging, which further resolves sign conflicts between task vectors. After random dropping, the TIES sign-conflict resolution is applied.

```python
DareTiesMerging(
    sparsity_ratio=0.5,
    only_on_linear_weights=False,
    rescale=True,
    scaling_factor=0.5,
    threshold=20,
    remove_keys=[],
    merge_func="sum",         # 'sum', 'mean', or 'max'
)
```

## Utility Functions

The DARE module provides these key utility functions in `fusion_bench/method/dare/utils.py`:

- **`param_random_drop_(param, sparsity_level, rescale)`**: Applies random drop-and-rescale to a single parameter tensor in-place.
- **`module_random_drop_(tv, sparsity_level, rescale)`**: Applies random drop to all parameters in a module or state dictionary.
- **`module_sub_(a, b, trainable_only=True)`**: Computes the difference between two models' parameters.
- **`trainable_state_dict(module)`**: Extracts only the trainable parameters from a module.

## Examples

### CLI Usage

Configuration template for DARE + Task Arithmetic:

```yaml title="config/method/dare/task_arithmetic.yaml"
--8<-- "config/method/dare/task_arithmetic.yaml"
```

Run DARE + Task Arithmetic:

```bash
fusion_bench method=dare/task_arithmetic \
    method.sparsity_ratio=0.5 \
    method.scaling_factor=0.3 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Configuration for DARE + Simple Average:

```yaml title="config/method/dare/simple_average.yaml"
--8<-- "config/method/dare/simple_average.yaml"
```

Run DARE + Simple Average:

```bash
fusion_bench method=dare/simple_average \
    method.sparsity_ratio=0.5 \
    method.rescale=true \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Configuration for DARE + TIES-Merging:

```yaml title="config/method/dare/ties_merging.yaml"
--8<-- "config/method/dare/ties_merging.yaml"
```

Run DARE + TIES-Merging:

```bash
fusion_bench method=dare/ties_merging \
    method.sparsity_ratio=0.5 \
    method.scaling_factor=0.5 \
    method.threshold=20 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method.dare import DareTaskArithmetic

# Instantiate the algorithm
algorithm = DareTaskArithmetic(
    scaling_factor=0.3,
    sparsity_ratio=0.5,
    only_on_linear_weights=False,
    rescale=True,
)

# Run the algorithm on a model pool
# The model pool must contain a pretrained model (_pretrained_) and fine-tuned models
merged_model = algorithm.run(modelpool)
```

## Implementation Details

The DARE implementation follows these steps in `DareTaskArithmetic.run()`:

1. Load the pretrained model from the model pool.
2. Compute task vectors $\tau_i = \theta_i - \theta_0$ for each fine-tuned model.
3. For each task vector, apply random drop-and-rescale (either on all parameters or only on `nn.Linear` weight matrices).
4. Sum the pruned task vectors.
5. Scale the summed task vector by `scaling_factor` and add to the pretrained model.

Key implementation notes:
- The random drop is performed in-place on the task vector parameters.
- When `only_on_linear_weights=True`, only the `.weight` attribute of `nn.Linear` modules is pruned; biases and non-linear parameters (e.g., LayerNorm weights) are left untouched.
- The `rescale` parameter controls whether the remaining elements are multiplied by $1/(1-p)$. Setting `rescale=False` disables this, which changes the semantics of the merge.

## Hyperparameter Guidelines

- **`sparsity_ratio`**: Typically set to 0.5. Higher values increase sparsity, reducing overlap between models but potentially losing more information.
- **`scaling_factor`**: Controls the strength of the merged task vector. Values between 0.1 and 1.0 are common; tune on a validation set.
- **`only_on_linear_weights`**: Set to `True` for language models (recommended by the original paper). Set to `False` when merging vision models or when all parameters are equally task-relevant.

## Implementation Details

- [DareSimpleAverage][fusion_bench.method.dare.DareSimpleAverage]
- [DareTaskArithmetic][fusion_bench.method.dare.DareTaskArithmetic]
- [DareTiesMerging][fusion_bench.method.dare.DareTiesMerging]

[^1]: (ICLR 2024) Yu et al. Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch. https://arxiv.org/abs/2311.03099
