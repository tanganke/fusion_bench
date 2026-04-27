# RankOne MoE for Model Merging

RankOne-MoE (Mixture of Experts) is a model merging approach that upscales merged models into a Mixture-of-Experts architecture. Instead of computing a single set of merged weights, it creates a pool of "experts" -- low-rank (rank-1) perturbations derived from each task-specific model -- and a learned router that dynamically selects which experts to use for each input.

**The Rank-One Expert Construction**. For each task-specific model, the algorithm computes a rank-one decomposition of the task-specific weight update. Given a task model with weight $W_i$ and the pretrained weight $W_0$, the update $\Delta W_i = W_i - W_0$ is approximated by:

$$\Delta W_i \approx a_i b_i^T$$

where $a_i$ and $b_i$ are vectors. Each such rank-one pair $(a_i, b_i)$ forms an "expert" for that task at that layer.

**Model Architecture**. The RankOne-MoE model replaces the MLP modules in each transformer layer with a `RankOneMoE` module. This module contains:
- A **base model** (the merged model via task arithmetic, made non-trainable).
- A **pool of experts** (rank-one perturbations, with `rank_k` experts per task).
- A **router** (a small network with `router_hidden_layers` hidden layers) that scores each expert and selects the top-$k$.

**The MoE Forward Pass**. For each input token, the router computes:

$$\text{output} = f_{\text{base}}(x) + \sum_{e \in \text{top-}k} \text{softmax}(s_e(x)) \cdot e(x)$$

where $s_e(x)$ is the router score for expert $e$ on input $x$, and $f_{\text{base}}$ is the base merged model.

## Test-Time Adaptation

After constructing the MoE model, the router parameters are fine-tuned via test-time adaptation. The optimization objective is **entropy maximization**:

$$\mathcal{L} = -\mathbb{E}_x \left[ \sum_c p(c|x) \log p(c|x) \right]$$

where $p(c|x)$ is the softmax probability of class $c$ given input $x$. Maximizing entropy encourages the router to produce confident, well-distributed predictions. The adaptation supports gradient accumulation across tasks (`use_grad_accumulate=true`) for memory efficiency.

## Examples

### CLI Usage

```yaml title="config/method/rankone_moe/rankone_moe.yaml"
--8<-- "config/method/rankone_moe/rankone_moe.yaml"
```

```bash
fusion_bench \
  method=rankone_moe/rankone_moe \
  method.rank_k=32 \
  method.select_k=-1 \
  method.init_lambda=0.3 \
  method.lr=0.0001 \
  method.max_steps=1000 \
  method.batch_size=16 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank_k` | int | 32 | Number of rank-one experts added per task to the expert pool. |
| `select_k` | int | -1 | Number of experts selected from the pool during inference. -1 means all experts. |
| `init_lambda` | float | 0.3 | Scaling factor for the initial task arithmetic merge (base model). |
| `lr` | float | 1e-4 | Learning rate for router test-time adaptation. |
| `max_steps` | int | 1000 | Number of test-time adaptation steps. |
| `batch_size` | int | 16 | Batch size for test-time adaptation. |
| `num_workers` | int | 16 | DataLoader workers. |
| `router_hidden_layers` | int | 1 | Number of hidden layers in the router network. |
| `batch_reduce` | bool | true | Whether to use batch-reduce mode. Set to false for sample-wise adaptation at inference. |
| `use_grad_accumulate` | bool | true | Use gradient accumulation across tasks to save memory. |
| `svd_accelerator` | str | "cuda" | Device for SVD computation during expert construction. |
| `checkpoint` | str/bool | False | Path to load a checkpoint (skips test-time adaptation). |
| `save_checkpoint` | str/bool | False | Path to save a checkpoint after adaptation. |

### API Usage

```python
from fusion_bench.method.rankone_moe import CLIPRankOneMoEAlgorithm

# The algorithm is configured via DictConfig in the YAML
# See config/method/rankone_moe/rankone_moe.yaml for the full config
moe_model = algorithm.run(modelpool)

# After construction, each layer's MLP is a RankOneMoE module
# with learnable router parameters
```

## Output

The algorithm returns a `RankOneMoE` model where each transformer layer's MLP has been replaced with a `RankOneMoE` module. The model supports both batch-reduced inference (all tasks processed together) and sample-wise inference (individual predictions).

## Implementation Details

- [fusion_bench.method.rankone_moe.rankone_moe.RankOneMoEAlgorithm][]
- [fusion_bench.method.rankone_moe.clip_rankone_moe.CLIPRankOneMoEAlgorithm][]
- [fusion_bench.models.rankone_moe.RankOneMoE][]

[^1]: (2024) RankOne-MoE: Mixture of Experts for Multi-Task Model Merging. https://github.com/EnnengYang/RankOne-MoE
