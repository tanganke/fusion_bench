# SparseLo Pruning

SparseLo (LoSparse) is a structured pruning framework for Llama models that combines magnitude-based weight pruning with low-rank compensation. The core idea is to prune a large fraction of weights while compensating for the lost information by learning a low-rank decomposition of the pruned weights.

**The LoSparse Linear Layer**. Each linear layer in the model is converted to a `LoSparseLinear` module, which consists of two parts:

$$y = W_{\text{sparse}} x + B \cdot (A \cdot x)$$

where:
- $W_{\text{sparse}}$ is the pruned (sparse) weight matrix.
- $B \in \mathbb{R}^{m \times r}$ and $A \in \mathbb{R}^{r \times n}$ form a low-rank (rank $r$) compensation matrix $B A^T$.

The low-rank part captures the information lost during pruning, preserving model accuracy with minimal additional parameters.

**Pruning Variants**. SparseLo supports multiple pruning strategies:

1. **`wanda`**: Activation-aware pruning using the WANDA method. Importance scores are computed as $|w_i| \cdot \max_j |x_j|$ where $x_j$ are calibration activations.

2. **`magnitude`**: Standard magnitude-based pruning using absolute weight values $|w_i|$ as importance scores.

3. **`random`**: Random pruning as a baseline.

4. **`lowrank-only`**: Extracts the low-rank part via SVD of the original weights and sets the sparse part to zero.

5. **`dense`**: No pruning (debug mode).

**Low-Rank Extraction**. After pruning, the low-rank compensation is extracted from the pruned weight matrix via SVD:

$$W_{\text{pruned}} = U \Sigma V^T \implies A = V_k^T, \quad B = U_k \Sigma_k$$

where $k = \text{rank}$. The sparse weight is then updated to $W_{\text{sparse}} = W_{\text{original}} - B A^T$.

## Advanced Variants

### PCP SparseLo (`PCPSparseLoForLlama`)

The PCP (Principal Component Pursuit) variant applies an optimization-based refinement step after pruning. Given the original weight $W$ and the pruning mask, it solves:

$$\min_q \ \|W \odot (1-m) + q \odot m\|_* + \lambda \|W \odot m - q \odot m\|_1$$

where $\| \cdot \|$ denotes the nuclear norm and $\| \cdot \|_1$ is the entrywise L1 norm, $\lambda = 1/\sqrt{\max(|W|)}$, and $m$ is the pruning mask. This is optimized via Adam for `num_iterations` steps with cosine annealing. The refined sparse weights replace the magnitude-pruned ones.

### Iterative SparseLo (`IterativeSparseLoForLlama`)

The iterative variant progressively reconstructs the pruned weight matrix. For each iteration with increasing rank $r$:

1. Compute the difference $W_{\text{diff}} = W_{\text{original}} - W_{\text{current}}$.
2. Perform SVD of $W_{\text{diff}}$, retaining only the tail components (rank $r$ and above).
3. Update: $W_{\text{current}} = W_{\text{current}} + \text{mask} \odots (U_{\text{tail}} \Sigma_{\text{tail}} V_{\text{tail}}^T)$.
4. Stop early if the spectrum ratio (fraction of singular value energy in the first $r$ components) exceeds 0.99.

This iterative reconstruction can optionally use a reference model's weights as the original.

## Examples

### CLI Usage -- Standard SparseLo

```yaml title="config/method/sparselo_pruning/llama_sparselo.yaml"
--8<-- "config/method/sparselo_pruning/llama_sparselo.yaml"
```

```bash
fusion_bench \
  method=sparselo_pruning/llama_sparselo \
  method.rank=128 \
  method.sparsity_ratio=0.5 \
  method.prune_type=unstructured \
  method.variant=wanda \
  method.nsamples=128 \
  modelpool=CausalLMPool/meta-llama/Llama-2-7b-hf
```

### CLI Usage -- Iterative SparseLo

```yaml title="config/method/sparselo_pruning/llama_iterative_sparselo.yaml"
--8<-- "config/method/sparselo_pruning/llama_iterative_sparselo.yaml"
```

```bash
fusion_bench \
  method=sparselo_pruning/llama_iterative_sparselo \
  method.rank=128 \
  method.num_iterations=10 \
  method.sparsity_ratio=0.5 \
  method.variant=wanda \
  method.use_reference_model=false \
  modelpool=CausalLMPool/meta-llama/Llama-2-7b-hf
```

### CLI Usage -- PCP SparseLo

```yaml title="config/method/sparselo_pruning/llama_pcp_sparselo.yaml"
--8<-- "config/method/sparselo_pruning/llama_pcp_sparselo.yaml"
```

```bash
fusion_bench \
  method=sparselo_pruning/llama_pcp_sparselo \
  method.rank=128 \
  method.num_iterations=10 \
  method.sparsity_ratio=0.5 \
  method.variant=wanda \
  modelpool=CausalLMPool/meta-llama/Llama-2-7b-hf
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` | int | 128 | Rank of the low-rank compensation matrix. |
| `variant` | str | "wanda" | Pruning variant: "dense", "random", "wanda", "lowrank-only", "magnitude". |
| `nsamples` | int | 128 | Number of calibration samples (for wanda variant). |
| `sparsity_ratio` | float | 0.5 | Fraction of weights to prune (unstructured). |
| `prune_type` | str | "unstructured" | "unstructured" or "semistructured". |
| `n`, `m` | int | 2, 4 | n:m ratio for semistructured pruning. |
| `num_iterations` | int | 10 | (Iterative/PCP only) Number of refinement iterations. |
| `use_reference_model` | bool | false | (Iterative only) Use a reference model's weights. |
| `model_save_path` | str | null | Path to save the pruned model. |

### API Usage

```python
from fusion_bench.method.sparselo.sparselo import SparseLoForLlama

algorithm = SparseLoForLlama(
    nsamples=128,
    variant="wanda",
    seed=0,
    rank=128,
    sparsity_ratio=0.5,
    prune_type="unstructured",
    n=2,
    m=4,
)
pruned_model = algorithm.run(modelpool)
```

## Model Conversion

The algorithm converts a standard `LlamaForCausalLM` to a `LoSparseLlamaForCausalLM` by replacing all `nn.Linear` modules with `LoSparseLinear` modules. Each `LoSparseLinear` layer has three weight matrices: `weight` (sparse), `lo_A`, and `lo_B` (low-rank).

## Implementation Details

- [fusion_bench.method.sparselo.sparselo.SparseLoForLlama][]
- [fusion_bench.method.sparselo.sparselo.PCPSparseLoForLlama][]
- [fusion_bench.method.sparselo.sparselo.IterativeSparseLoForLlama][]
- [fusion_bench.models.modeling_losparse_llama.losparse_linear.LoSparseLinear][]

[^1]: SparseLo: Sparse Models with Low-Rank Compensation for Efficient Language Model Compression.
