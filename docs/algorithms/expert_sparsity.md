# Expert Sparsity

Expert Sparsity provides a suite of methods for pruning and optimizing Mixture-of-Experts (MoE) language models, specifically targeting Mixtral architectures. The goal is to reduce the number of experts or the computation per token while maintaining model quality, enabling faster inference and lower memory usage.

The implementation follows the paper "Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models" (ACL 2024), which presents three complementary techniques: Layer-Wise Pruning, Progressive Pruning, and Dynamic Skipping.

## Layer-Wise Pruning

Layer-wise pruning selects a subset of experts for each layer independently. The algorithm works as follows:

1. **Wrapper Insertion**: Each `MixtralSparseMoeBlock` is wrapped in a `PrunableMixtralSparseMoeBlockWrapper` that caches input activations ($X$) and intermediate outputs ($Z$).

2. **Calibration Forward Pass**: Calibration data is forwarded through the model to accumulate activation statistics.

3. **Expert Enumeration**: For each layer, the wrapper evaluates all possible combinations of preserving $r$ experts out of the total, computing the reconstruction loss (difference between the original MoE output and the pruned output) for each combination.

4. **Optimal Selection**: The combination with the lowest reconstruction loss is selected, and the pruned MoE block retains only those experts.

$$\text{loss}(S) = \frac{1}{|D|} \sum_{x \in D} \| \text{MoE}_{\text{full}}(x) - \text{MoE}_{\text{pruned}, S}(x) \|_2^2$$

where $S$ is a subset of $r$ experts.

### CLI Usage

```yaml title="config/method/expert_sparsity/mixtral.yaml"
--8<-- "config/method/expert_sparsity/mixtral.yaml"
```

```bash
fusion_bench \
  method=expert_sparsity/mixtral \
  method._target_=fusion_bench.method.LayerWisePruningForMixtral \
  method.num_preserved_experts=4 \
  method.calib_set=c4 \
  method.n_blocks_for_stat=128 \
  method.batch_size=8 \
  modelpool=CausalLMPool/mixtral-8x7b
```

## Progressive Pruning

Progressive pruning is a memory-efficient variant that prunes one layer at a time, replacing the wrapper with the pruned model before moving to the next layer. This reduces peak memory usage:

1. **Z-activation Collection**: First pass collects only the intermediate expert outputs ($Z$) for all layers.

2. **Layer-by-Layer X-Collection**: For each layer, a forward pass collects the input activations ($X$) for that layer only. After enumerating and pruning, the wrapper is replaced with the pruned model, freeing memory.

3. **Result**: The same optimal subset selection as layer-wise pruning, but with lower memory overhead.

### CLI Usage

```bash
fusion_bench \
  method=expert_sparsity/mixtral \
  method._target_=fusion_bench.method.ProgressivePruningForMixtral \
  method.num_preserved_experts=4 \
  method.calib_set=c4 \
  method.n_blocks_for_stat=128 \
  method.batch_size=8 \
  modelpool=CausalLMPool/mixtral-8x7b
```

## Dynamic Skipping

Dynamic skipping is a runtime optimization that analyzes the routing weight ratios across calibration data to determine per-layer beta parameters. These betas control how aggressively tokens can skip the second-ranked expert during inference:

1. **Router Logit Collection**: The wrapper caches router logits, input activations ($X$), and expert outputs ($Z$).

2. **Ratio Analysis**: For each token, the ratio of the second-highest routing weight to the highest is computed:

$$\rho = \frac{w_{(2)}}{w_{(1)}}$$

where $w_{(1)}$ and $w_{(2)}$ are the sorted routing weights (descending).

3. **Beta Computation**: The median (and mean) of $\rho$ across all tokens and positions is computed per layer. The median is stored as `model.config.betas[layer_idx]` and used at inference time to decide whether the second expert can be skipped.

### CLI Usage

```bash
fusion_bench \
  method=expert_sparsity/mixtral \
  method._target_=fusion_bench.method.DynamicSkippingPruningForMixtral \
  method.calib_set=c4 \
  method.n_blocks_for_stat=128 \
  method.batch_size=8 \
  modelpool=CausalLMPool/mixtral-8x7b
```

## Calibration Data

All three methods use calibration data for analysis. Supported datasets:

- **C4**: The Common Crawl C4 corpus (English subset). Downloaded from `allenai/c4` on HuggingFace Hub.
- **MATH**: A math pretraining-style dataset from `tanganke/math_pretrain_style`.

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_preserved_experts` | int | 4 | Number of experts to keep per layer (pruning methods). |
| `calib_set` | str | "c4" | Calibration dataset: "c4" or "math". |
| `max_block_size` | int | 2048 | Max sequence length per calibration sample. |
| `n_blocks_for_stat` | int | 128 | Number of sequence blocks for calibration. 0 = use entire dataset. |
| `batch_size` | int | 8 | Batch size for calibration forward passes. |
| `num_workers` | int | 8 | DataLoader workers. |
| `seed` | int | 42 | Random seed for calibration data shuffling. |
| `model_save_path` | str | "{log_dir}/pruned_model" | Path to save the pruned model. |

## Output

- **Pruning methods**: Return the pruned `MixtralForCausalLM` model with `num_experts` reduced to `num_preserved_experts`. Also save pruning info (loss history per layer) to `{log_dir}/pruning_info.pt`.

- **Dynamic Skipping**: Return the original model with `config.betas` set to the per-layer median routing ratios. Also save `(res_median, res_mean)` to `{log_dir}/pruning_info.pt`.

## Implementation Details

- [fusion_bench.method.expert_sparsity.mixtral.layer_wise_pruning.LayerWisePruningForMixtral][]
- [fusion_bench.method.expert_sparsity.mixtral.layer_wise_pruning.layerwise_pruning][]
- [fusion_bench.method.expert_sparsity.mixtral.progressive_pruning.ProgressivePruningForMixtral][]
- [fusion_bench.method.expert_sparsity.mixtral.progressive_pruning.progressive_pruning][]
- [fusion_bench.method.expert_sparsity.mixtral.dynamic_skipping.DynamicSkippingPruningForMixtral][]
- [fusion_bench.method.expert_sparsity.mixtral.dynamic_skipping.dynamic_skipping][]
- [fusion_bench.method.expert_sparsity.utils.calibration_data.build_calib_loader][]

[^1]: (ACL 2024) Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models. http://arxiv.org/abs/2402.14800
[^2]: Original repo: https://github.com/Lucky-Lance/Expert_Sparsity
