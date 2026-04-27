# RandES (Random Erasure Superposition)

RandES is a model superposition and retrieval framework that compresses multiple fine-tuned models into a single superposed representation, from which individual models can be later retrieved. By applying random linear transformations (erasure matrices) to each model's parameters and superposing the transformed parameters, RandES enables storing $K$ models in approximately the space of one[^1].

<figure markdown="span">
  ![RandES](images/randes.png){ width="600" }
  <figcaption>RandES: Random erasure and superposition for compact multi-model storage. Credit to <sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></figcaption>
</figure>

**Core Idea**. Each model's parameters are transformed by a model-specific random matrix, then all transformed parameters are summed into a single superposed tensor. To retrieve a specific model, the superposed tensor is multiplied by the retrieval matrix (the inverse or pseudo-inverse of the model's erasure matrix). The randomization ensures that the superposition does not cause catastrophic interference.

## Mathematical Formulation

Given $K$ models with parameters $\{\theta_i\}_{i=1}^K$ and a pretrained model $\theta_0$:

### Compression Phase

For each target layer with parameter matrix $W_i \in \mathbb{R}^{d_{out} \times d_{in}}$ of model $i$, apply a random transformation matrix $M_i$:

$$\tilde{W}_i = W_i M_i$$

The superposed layer is:

$$W_{superposed} = \sum_{i=1}^{K} \tilde{W}_i = \sum_{i=1}^{K} W_i M_i$$

### Retrieval Phase

To retrieve model $j$, apply the retrieval matrix $R_j = M_j^{-1}$ (or $M_j^\top$ for orthogonal matrices):

$$\hat{W}_j = W_{superposed} R_j = \sum_{i=1}^{K} W_i M_i M_j^{-1}$$

When $M_i$ are independent random orthogonal/diagonal matrices, the cross-terms $W_i M_i M_j^{-1}$ for $i \neq j$ become noise that averages out, while the desired term $W_j$ is recovered.

## Erasure Matrix Modes

RandES supports multiple erasure matrix modes, each with different properties:

### 1. Random Binary Diagonal Matrix (`random_binary_diagonal_matrix`)

Each model $i$ gets a diagonal matrix with entries $\pm 1$, generated from a Bernoulli distribution with $p=0.5$:

$$M_i = \text{diag}(s_{i,1}, s_{i,2}, \dots, s_{i,d_{in}}), \quad s_{i,j} \in \{-1, +1\}$$

Compression: element-wise Hadamard product $\tilde{W}_i = W_i \odot M_i$.
Retrieval: $\hat{W}_i = W_{superposed} \odot M_i$ (since $M_i^{-1} = M_i$ for $\pm 1$ diagonal).

This mode is highly efficient and uses only 1 byte per element for context storage (the sign bits).

### 2. Random Rotation Matrix (`random_rotation_matrix`)

Each model gets an orthogonal rotation matrix sampled from the orthogonal group (using `scipy.stats.ortho_group`):

$$M_i \in \mathbb{R}^{d_{in} \times d_{in}}, \quad M_i^\top M_i = I$$

Compression: matrix multiplication $\tilde{W}_i = W_i M_i$.
Retrieval: $\hat{W}_i = W_{superposed} M_i^\top$ (since $M_i^{-1} = M_i^\top$).

### 3. Random Dense Matrix (`random_dense_matrix`)

Each model gets a random Gaussian matrix. The pseudo-inverse is used for retrieval:

$$M_i \sim \mathcal{N}(0, I)^{d_{in} \times d_{in}}$$
$$R_i = M_i^+ = (M_i^\top M_i)^{-1} M_i^\top$$

### 4. Random Diagonal Matrix (`random_diagonal_matrix`)

Each model gets a diagonal matrix with Gaussian entries. Retrieval uses element-wise division:

$$M_i = \text{diag}(g_{i,1}, g_{i,2}, \dots), \quad g_{i,j} \sim \mathcal{N}(0, 1)$$
$$R_i = \text{diag}(1/g_{i,1}, 1/g_{i,2}, \dots)$$

### 5. Identity Matrix (`identity_matrix`)

The identity matrix -- no transformation. This serves as a baseline (simple superposition without randomization).

## Layer Shifting

RandES supports layer shifting, where each model's layers are permuted before superposition:

- **Fixed shift** (`shift_layers > 0`): Each model $i$ has its layers cyclically shifted by $i \times \text{shift\_layers}$ positions.
- **Random shift** (`shift_layers = -1`): Each model gets a random permutation of its layers.

Layer shifting further decorrelates the superposed representations of different models.

## Absorber

Non-target layers (e.g., LayerNorm, embeddings) can be "absorbed" rather than superposed. The absorber options are:

- **`average`**: Use the average of all models' non-target layers.
- **`pretrained`**: Use the pretrained model's non-target layers.
- **`None`**: No absorption -- all layers are superposed.

## RandES Variants in FusionBench

### 1. Superposed Model Soup

The `SuperposedModelSoupAlgorithm` compresses full model weights into a superposed representation. Models are retrieved as individual weights.

```python
from fusion_bench.method.randes import SuperposedModelSoupAlgorithm

algorithm = SuperposedModelSoupAlgorithm(
    mode="random_binary_diagonal_matrix",
    target_layer=["mlp_w", "attn_w"],
    random_seed=42,
    different_across_layers=True,
    shift_layers=0,
    absorber="average",
    ms_mode="average",     # "average" or "original"
    dropout_rate=1,        # process every Nth target layer
)

result = algorithm.run(modelpool)
# result = {"models": {model_name: retrieved_model, ...}, "metadata": {...}}
```

### 2. Superposed Task Arithmetic

The `SuperposedTaskArithmeticAlgorithm` compresses task vectors (differences from pretrained) rather than full weights. This is more efficient because the pretrained model serves as an anchor and does not need to be stored in the superposed representation.

$$\tau_i = \theta_i - \theta_0$$
$$W_{superposed} = \sum_i \tau_i M_i$$

Retrieval yields the task vector, which is then added back to the pretrained model:

$$\hat{\theta}_i = \theta_0 + \lambda \cdot \text{retrieve}(W_{superposed}, M_i)$$

```python
from fusion_bench.method.randes import SuperposedTaskArithmeticAlgorithm

algorithm = SuperposedTaskArithmeticAlgorithm(
    mode="random_binary_diagonal_matrix",
    target_layer=["mlp_w", "attn_w"],
    random_seed=42,
    scaling_factor=0.5,
    model_path=None,
)
```

### 3. Superposed Task Arithmetic with LoRA

The `SuperposedTaskArithmeticLoRAAlgorithm` works with LoRA adapters. It compresses the LoRA weight matrices ($A$ and $B$), and during retrieval, reconstructs the merged LoRA weight as $W_{lora} = B \cdot A$ before adding to the pretrained model.

## Common Configuration

All RandES variants share these base parameters defined in `SuperposedAlgorithmBase`:

| Parameter | Description |
|---|---|
| `mode` | Erasure matrix type |
| `target_layer` | Layers to apply superposition to (e.g., `["mlp_w", "attn_w"]`) |
| `random_seed` | Seed for random matrix generation |
| `different_across_layers` | Use different matrices per layer |
| `rank` | Rank for SVD-based decomposition |
| `random_components` | Use random SVD components |
| `shift_layers` | Layer shift amount (0 = none, -1 = random) |
| `absorber` | How to handle non-target layers |
| `debug` | Debug level (0-2) |
| `ms_mode` | For model soup: "average" or "original" |
| `verbose` | Verbosity level |
| `dropout_rate` | Process every Nth target layer |

### Target Layer Filtering

The `target_layer` parameter accepts these values:
- `"mlp_w"`: MLP weight matrices (excluding biases)
- `"attn_w"`: Attention weight matrices (excluding biases)
- `"mlp"`: All MLP parameters (including biases)
- `"attn"`: All attention parameters (including biases)
- `"all"`: All parameters

### Dropout Rate

The `dropout_rate` parameter controls sparsity of the target layers. If `dropout_rate=2`, only every 2nd target layer is compressed; the rest use the absorber. This reduces storage while maintaining retrieval quality.

## Analysis Utilities

RandES provides comprehensive analysis utilities in `fusion_bench/method/randes/base_algorithm.py`:

- **`cosine_similarity(tensor1, tensor2)`**: Compute cosine similarity between two tensors.
- **`svd_and_partition(A, num_chunks)`**: Partition SVD into chunks for subspace analysis.
- **`compute_svd_subspace_similarity(ref, retrieval, num_chunks)`**: Evaluate how well retrieved weights match the original in SVD subspace.
- **`pairwise_cosine_similarity_matrix(tensors)`**: Compute all-pairs cosine similarity.
- **`compare_models(state_dict1, state_dict2)`**: Comprehensive comparison including L2 differences and cosine similarities at layer and global levels.

## Examples

### CLI Usage

Configuration for Superposed Model Soup:

```yaml title="config/method/randes/superposed_model_soup.yaml"
--8<-- "config/method/randes/superposed_model_soup.yaml"
```

Run Superposed Model Soup:

```bash
fusion_bench method=randes/superposed_model_soup \
    method.mode=random_binary_diagonal_matrix \
    method.target_layer=mlp_w,attn_w \
    method.random_seed=42 \
    method.absorber=average \
    method.ms_mode=average \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=...
```

Configuration for Superposed Task Arithmetic:

```yaml title="config/method/randes/superposed_task_arithmetic.yaml"
--8<-- "config/method/randes/superposed_task_arithmetic.yaml"
```

Run Superposed Task Arithmetic:

```bash
fusion_bench method=randes/superposed_task_arithmetic \
    method.mode=random_binary_diagonal_matrix \
    method.scaling_factor=0.5 \
    method.shift_layers=0 \
    method.absorber=pretrained \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=...
```

Configuration for Superposed Task Arithmetic with LoRA:

```yaml title="config/method/randes/superposed_task_arithmetic_lora.yaml"
--8<-- "config/method/randes/superposed_task_arithmetic_lora.yaml"
```

Run Superposed Task Arithmetic LoRA:

```bash
fusion_bench method=randes/superposed_task_arithmetic_lora \
    method.mode=random_binary_diagonal_matrix \
    method.scaling_factor=0.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_lora \
    taskpool=...
```

### API Usage

```python
from fusion_bench.method.randes import SuperposedTaskArithmeticAlgorithm

algorithm = SuperposedTaskArithmeticAlgorithm(
    mode="random_binary_diagonal_matrix",
    target_layer=["mlp_w", "attn_w"],
    random_seed=42,
    different_across_layers=True,
    shift_layers=0,
    absorber="pretrained",
    scaling_factor=0.5,
    model_path="outputs/retrieved_models.pt",
)

result = algorithm.run(modelpool)
models = result["models"]  # dict of {model_name: retrieved_model}
metadata = result["metadata"]  # dict with analysis info
```

## Storage Efficiency

One of the key benefits of RandES is storage efficiency. The metadata returned includes:

- **`total_gb_original`**: Total storage of the original models (in GB).
- **`total_gb_retrieved`**: Total storage of the superposed + context representation.
- **`nonzero_parameter_count`**: Number of non-zero parameters in the superposed representation.
- **`nonzero_param_count_context`**: Number of non-zero parameters in the context (retrieval matrices).

For the `random_binary_diagonal_matrix` mode, context storage is minimal (1 byte per element for the sign bits), making it highly storage-efficient.

## Hyperparameter Guidelines

- **`mode`**: Use `random_binary_diagonal_matrix` for best storage efficiency. Use `random_rotation_matrix` for better retrieval quality at the cost of larger context.
- **`target_layer`**: Focus on MLP and attention weights (`["mlp_w", "attn_w"]`) as they contain the bulk of task-specific knowledge.
- **`shift_layers`**: Use `-1` (random shuffle) for maximum decorrelation. Use a fixed value for controlled experiments.
- **`dropout_rate`**: Higher values (e.g., 2 or 3) skip more layers, reducing storage but potentially degrading retrieval quality.
- **`absorber`**: Use `"pretrained"` when non-target layers were frozen during fine-tuning. Use `"average"` for a compromise.

## Implementation Details

- [SuperposedAlgorithmBase][fusion_bench.method.randes.SuperposedAlgorithmBase]
- [SuperposedModelSoupAlgorithm][fusion_bench.method.randes.SuperposedModelSoupAlgorithm]
- [SuperposedTaskArithmeticAlgorithm][fusion_bench.method.randes.SuperposedTaskArithmeticAlgorithm]
- [SuperposedTaskArithmeticLoRAAlgorithm][fusion_bench.method.randes.SuperposedTaskArithmeticLoRAAlgorithm]

[^1]: (2024) Random Erasure Superposition for Compact Multi-Model Storage. The implementation follows the principles of randomized superposition for model compression and retrieval.
