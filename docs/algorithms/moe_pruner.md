# MoE Pruner

MoE Pruner applies post-training magnitude pruning to Mixture-of-Experts (MoE) language models. It supports both Mixtral and DeepSeek V2 architectures, pruning the linear layers within expert modules based on importance scores derived from calibration data.

**The Pruning Strategy**. MoE Pruner uses a layer-by-layer pruning approach with activation-based importance scoring:

1. **Calibration Data Preparation**: The algorithm loads calibration data (from the C4 dataset) and propagates it through the model layer by layer, capturing the inputs to each layer.

2. **Importance Score Computation**: For each layer, forward hooks are registered on all linear layers within the expert modules. A gate-level hook captures which experts are activated for each token (via the MoE routing mechanism). The importance of each weight is computed as the product of the absolute weight value and the accumulated activation magnitude.

3. **Magnitude Pruning**: Weights are pruned based on their importance scores. The least important weights (lowest scores) are set to zero.

**Supported Pruning Types**:
- **Unstructured pruning**: A fraction `sparsity_ratio` of weights are pruned (e.g., 0.5 = 50% sparsity).
- **Semistructured pruning**: Weights are pruned in an n:m pattern (e.g., 2:4 means 2 out of every 4 consecutive weights are pruned).

**Architecture-Specific Hooks**: The implementation provides specialized hook functions for both Mixtral and DeepSeek V2 models:

- **Mixtral**: `MoEPrunerHookFnForMixtralLinear` and `MoEPrunerHookFnForMixtralGate`
- **DeepSeek V2**: `MoEPrunerHookFnForDeepseekV2Linear` and `MoEPrunerHookFnForDeepseekV2Gate`

For DeepSeek V2, the algorithm handles both MoE layers (with expert modules) and dense MLP layers, skipping the latter without pruning.

## Layer-by-Layer Processing

The pruning proceeds layer by layer to manage memory:

1. The inputs to the current layer are moved to the layer's device (supporting device mapping for large models).
2. Forward hooks are registered on all expert linear layers and the gate.
3. Calibration samples are forwarded through the layer.
4. Importance scores are computed and hooks removed.
5. Pruning is applied based on the scores.
6. The layer's outputs become the inputs for the next layer.

## Examples

### CLI Usage

```yaml title="config/method/moe_pruner/moe_pruner.yaml"
--8<-- "config/method/moe_pruner/moe_pruner.yaml"
```

```bash
fusion_bench \
  method=moe_pruner/moe_pruner \
  method.sparsity_ratio=0.5 \
  method.prune_type=unstructured \
  method.nsamples=100 \
  method.max_seqlen=2048 \
  modelpool=CausalLMPool/mixtral-8x7b
```

### Semistructured Pruning

For n:m structured sparsity (e.g., compatible with NVIDIA Tensor Cores):

```bash
fusion_bench \
  method=moe_pruner/moe_pruner \
  method.prune_type=semistructured \
  method.n=2 \
  method.m=4 \
  modelpool=CausalLMPool/mixtral-8x7b
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nsamples` | int | 100 | Number of calibration samples. |
| `seed` | int | 42 | Random seed for calibration data selection. |
| `device` | str | "cuda" | Device for computation. |
| `max_seqlen` | int | 2048 | Maximum sequence length for calibration data. |
| `prune_type` | str | "unstructured" | Pruning type: "unstructured" or "semistructured". |
| `sparsity_ratio` | float | 0.5 | Fraction of weights to prune (unstructured only). |
| `n` | int | 2 | N in n:m semistructured pruning. |
| `m` | int | 4 | M in n:m semistructured pruning. |

### API Usage

```python
from fusion_bench.method.moe_pruner import MoEPruner

algorithm = MoEPruner(
    nsamples=100,
    seed=42,
    device="cuda",
    max_seqlen=2048,
    prune_type="unstructured",
    sparsity_ratio=0.5,
    n=2,
    m=4,
)
pruned_model = algorithm.run(modelpool)
```

## Calibration Data Caching

Calibration data preparation is automatically cached to disk (`outputs/cache/{model_name}/calibration_data.pkl`) to avoid reprocessing on subsequent runs.

## Supported Models

- **MixtralForCausalLM** (e.g., Mixtral-8x7B)
- **DeepseekV2ForCausalLM** (e.g., DeepSeek V2/V2.5)

## Implementation Details

- [fusion_bench.method.moe_pruner.moe_pruner.MoEPruner][]
- [fusion_bench.method.moe_pruner.hooks.mixtral.MoEPrunerHookFnForMixtralGate][]
- [fusion_bench.method.moe_pruner.hooks.mixtral.MoEPrunerHookFnForMixtralLinear][]
- [fusion_bench.method.moe_pruner.hooks.deepseek_v2.MoEPrunerHookFnForDeepseekV2Gate][]
- [fusion_bench.method.moe_pruner.hooks.deepseek_v2.MoEPrunerHookFnForDeepseekV2Linear][]

[^1]: Adapted from the WANDA pruning methodology: https://github.com/locuslab/wanda
