# LM Fine-tuning

FusionBench provides three fine-tuning methods for language models: Full Fine-tuning for Supervised Fine-Tuning (SFT), PEFT (LoRA) Fine-tuning for SFT, and Bradley-Terry Reward Modeling. These methods use PyTorch Lightning Fabric for distributed training and support FSDP, gradient accumulation, and configurable checkpointing.

## Full Fine-tuning SFT

The `FullFinetuneSFT` algorithm performs full-parameter fine-tuning of a causal language model on supervised instruction datasets. All parameters of the model are updated (optionally excluding token embeddings via `fix_token_embedding`).

**Training Loop**. For each batch, the model computes the autoregressive language modeling loss:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t} \log p(y_t | y_{<t}, x; \theta)$$

where $N$ is the number of samples and the inner sum is over token positions. The loss is computed via the model's built-in cross-entropy (with labels shifted by one position).

**Key Features**:
- Supports gradient accumulation (`accumulate_grad_batches`).
- Gradient clipping by value or norm (`gradient_clip_val`, `gradient_clip_algorithm`).
- Configurable checkpointing (epoch or step interval, lightning or HuggingFace format).
- FSDP-compatible with gradient checkpointing.
- Optional token embedding freezing (`fix_token_embedding=true`).

### CLI Usage

```yaml title="config/method/lm_finetune/fullfinetune_sft.yaml"
--8<-- "config/method/lm_finetune/fullfinetune_sft.yaml"
```

```bash
fusion_bench \
  method=lm_finetune/fullfinetune_sft \
  method.optimizer.lr=1e-5 \
  method.max_epochs=3 \
  method.dataloader_kwargs.batch_size=1 \
  method.max_length=4096 \
  method.fix_token_embedding=true \
  modelpool=CausalLMPool/meta-llama/Llama-2-7b-hf \
  taskpool=dummy
```

## PEFT Fine-tuning SFT

The `PeftFinetuneSFT` algorithm applies Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters. Only the LoRA parameters are updated, keeping the base model frozen.

**LoRA Configuration**. LoRA low-rank adapters are applied to specified modules:

$$W(x) = W_0 x + \frac{1}{\alpha} B A x$$

where $W_0$ is the frozen original weight, $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ are the trainable low-rank matrices, $r$ is the LoRA rank, and $\alpha$ is the scaling factor (`lora_alpha`).

**Key Features**:
- Default targets: `q_proj`, `v_proj` (attention) and `gate_proj`, `down_proj`, `up_proj` (MLP).
- LoRA rank `r=64`, `lora_alpha=16`, `lora_dropout=0` (configurable).
- Post-training merge and unload option (`merge_and_unload=true`).
- Supports both Lightning and PEFT checkpoint formats.

### CLI Usage

```yaml title="config/method/lm_finetune/peftfinetune_sft.yaml"
--8<-- "config/method/lm_finetune/peftfinetune_sft.yaml"
```

```bash
fusion_bench \
  method=lm_finetune/peftfinetune_sft \
  method.peft_config.r=64 \
  method.peft_config.lora_alpha=16 \
  method.optimizer.lr=1e-4 \
  method.max_epochs=3 \
  method.merge_and_unload=false \
  modelpool=CausalLMPool/meta-llama/Llama-2-7b-hf \
  taskpool=dummy
```

## Bradley-Terry Reward Modeling

The `BradleyTerryRewardModeling` algorithm trains a reward model using the Bradley-Terry pairwise preference model. Given pairs of (chosen, rejected) responses, it learns to assign higher rewards to the chosen response.

**The Bradley-Terry Loss**:

$$\mathcal{L} = -\mathbb{E} \left[ \log \sigma(r_{\theta}(x, y_{\text{chosen}}) - r_{\theta}(x, y_{\text{rejected}})) \right]$$

where $r_{\theta}$ is the reward model (a sequence classification head on top of the LLM), $\sigma$ is the sigmoid function, and $(x, y_{\text{chosen}})$ and $(x, y_{\text{rejected}})$ form a preference pair.

**Dataset Format**. Each sample contains:
- `chosen_input_ids`, `chosen_attention_mask`: Token IDs for the preferred response.
- `rejected_input_ids`, `rejected_attention_mask`: Token IDs for the rejected response.

The collate function stacks chosen and rejected samples in a single batch (batch size must be even).

### CLI Usage

```yaml title="config/method/lm_finetune/bradley_terry_rm.yaml"
--8<-- "config/method/lm_finetune/bradley_terry_rm.yaml"
```

```bash
fusion_bench \
  method=lm_finetune/bradley_terry_rm \
  method.optimizer.lr=1e-5 \
  method.max_epochs=3 \
  method.dataloader_kwargs.batch_size=2 \
  method.max_length=4096 \
  modelpool=SequenceClassificationModelPool/reward_model \
  taskpool=dummy
```

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | int | 3 | Max training epochs (-1 = use max_steps). |
| `max_steps` | int | -1 | Max training steps (-1 = use max_epochs). |
| `max_steps_per_epoch` | int | -1 | Max steps per epoch. |
| `accumulate_grad_batches` | int | 1 | Gradient accumulation factor. |
| `gradient_clip_val` | float | null | Gradient clipping threshold. |
| `gradient_clip_algorithm` | str | "norm" | Clipping: "value" or "norm". |
| `checkpoint_save_interval` | str | "epoch" | "epoch" or "step". |
| `checkpoint_save_frequency` | int | 1 | Checkpoint frequency. |
| `save_ckpt_type` | str | "lightning" | "lightning", "hf", or "peft". |
| `save_full_model` | bool | true | Save full model or only trainable params. |
| `save_optimizer_state` | bool | false | Save optimizer state in checkpoint. |
| `ckpt_path` | str | null | Path to resume from checkpoint. |
| `max_length` | int | 4096 | Max sequence length. |
| `fix_token_embedding` | bool | true | Freeze token embeddings (SFT/RM only). |

## LR Scheduler Configuration

The `_T_max_` placeholder in LR scheduler configs is automatically replaced with the computed total number of training steps. This allows the scheduler to be configured without knowing the dataset size in advance.

## Implementation Details

- [fusion_bench.method.lm_finetune.fullfinetune_sft.FullFinetuneSFT][]
- [fusion_bench.method.lm_finetune.peftfinetune_sft.PeftFinetuneSFT][]
- [fusion_bench.method.lm_finetune.bradley_terry_rm.BradleyTerryRewardModeling][]

[^1]: The Bradley-Terry model for reward modeling follows the approach used in InstructGPT and RLHF pipelines.
