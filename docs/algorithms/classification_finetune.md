# Classification Fine-tuning

FusionBench provides comprehensive fine-tuning support for image classification across two major model families: CLIP-based vision models and generic image classification models (e.g., ResNet). These methods are used both as standalone training pipelines and as the fine-tuning step before model merging algorithms.

## CLIP Fine-tuning

The CLIP fine-tuning method trains the vision encoder of a CLIP model for zero-shot image classification. The classification is performed by computing cosine similarity between image features and text-derived class embeddings (zero-shot weights).

**Training Loop**. For each training step, batches from all tasks are fetched, and the cross-entropy loss is computed between the predicted logits and true labels:

$$\mathcal{L} = \sum_{t} \text{CE}\left(\text{logits}(x_t, \text{zeroshot}_t), y_t\right)$$

The logits are computed via the CLIP zero-shot classification pipeline:

$$\text{logits} = \frac{1}{\tau} \cdot f_{\text{vision}}(x) \cdot W_{\text{zeroshot}}^T$$

where $f_{\text{vision}}$ is the CLIP vision encoder, $W_{\text{zeroshot}}$ contains the text embeddings for class names, and $\tau$ is the logit scale.

**Supported Modes**:
- **Full fine-tuning**: All vision model parameters are updated.
- **LoRA fine-tuning**: Low-rank adapters are applied to the vision transformer's attention layers.
- **L-LoRA fine-tuning**: Linearized LoRA (from the PETA paper) for partial linearization of LoRA adapters.

### CLI Usage -- CLIP Fine-tune

```yaml title="config/method/classification/clip_finetune.yaml"
--8<-- "config/method/classification/clip_finetune.yaml"
```

```bash
fusion_bench \
  method=classification/clip_finetune \
  method.learning_rate=1e-5 \
  method.num_steps=4000 \
  method.batch_size=128 \
  method.use_lora=false \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_mtl \
  taskpool=dummy
```

### CLI Usage -- Continual CLIP Fine-tune

The continual fine-tuning variant trains on each task sequentially, resetting the optimizer and learning rate scheduler for each task.

```yaml title="config/method/classification/clip_continual_finetune.yaml"
--8<-- "config/method/classification/clip_continual_finetune.yaml"
```

```bash
fusion_bench \
  method=classification/clip_continual_finetune \
  method.shuffle_order=true \
  method.learning_rate=1e-5 \
  method.num_steps=4000 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_mtl \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### CLIP Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 1e-5 | Learning rate for the Adam optimizer. |
| `weight_decay` | float | 0 | Weight decay. |
| `num_steps` | int | 4000 | Number of training steps (per task for continual mode). |
| `batch_size` | int | 128 | Batch size per task. |
| `num_workers` | int | 16 | DataLoader workers. |
| `save_interval` | int | 500 | Checkpoint save frequency. |
| `use_lora` | bool | false | Enable LoRA adapters. |
| `use_l_lora` | bool | false | Enable L-LoRA (CLIP finetune only). |
| `shuffle_order` | bool | true | Shuffle task order (continual mode only). |
| `state_dict_load_path` | str | null | Path to resume training from. |
| `state_dict_save_path` | str | null | Path to save final model. |
| `skip_training` | bool | false | Skip training, only load and evaluate. |

## Generic Image Classification Fine-tuning

The generic image classification fine-tuning method uses PyTorch Lightning with the `lit-learn` library's `ERM_LitModule`. It supports any image classification model (ResNet, etc.) and uses standard cross-entropy loss with configurable label smoothing.

### CLI Usage

```yaml title="config/method/classification/image_classification_finetune.yaml"
--8<-- "config/method/classification/image_classification_finetune.yaml"
```

```bash
fusion_bench \
  method=classification/image_classification_finetune \
  method.max_epochs=10 \
  method.optimizer._target_=torch.optim.SGD \
  method.optimizer.lr=0.001 \
  method.dataloader_kwargs.batch_size=256 \
  modelpool=ResNetForImageClassificationPool/resnet50_imagenet \
  taskpool=dummy
```

### Generic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | int | 10 | Maximum training epochs. Mutually exclusive with `max_steps`. |
| `max_steps` | int | null | Maximum training steps. Mutually exclusive with `max_epochs`. |
| `label_smoothing` | float | 0 | Label smoothing factor for cross-entropy. |
| `save_top_k` | int | 1 | Number of best checkpoints to save. |
| `save_interval` | int | 1 | Checkpoint save interval. |
| `optimizer` | DictConfig | SGD | Optimizer configuration. |
| `lr_scheduler` | DictConfig | CosineAnnealingLR | LR scheduler configuration. |
| `dataloader_kwargs` | DictConfig | batch_size=256 | DataLoader configuration. |
| `training_data_ratio` | float | null | Fraction of training data to use. |

### API Usage

```python
from fusion_bench.method.classification import (
    ImageClassificationFineTuningForCLIP,
    ContinualImageClassificationFineTuningForCLIP,
    ImageClassificationFineTuning,
)

# CLIP fine-tuning
clip_algorithm = ImageClassificationFineTuningForCLIP(
    learning_rate=1e-5,
    num_steps=4000,
    batch_size=128,
)
finetuned_vision = clip_algorithm.run(modelpool)

# Generic image classification
generic_algorithm = ImageClassificationFineTuning(
    max_epochs=10,
    label_smoothing=0.0,
)
finetuned_model = generic_algorithm.run(modelpool)
```

## Multi-GPU Training

For CLIP fine-tuning on multiple GPUs:

```bash
fusion_bench \
  fabric.devices=8 \
  method=classification/clip_finetune \
  method.batch_size=2 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_mtl \
      modelpool.models.0.path=openai/clip-vit-large-patch14 \
  taskpool=dummy
```

## Implementation Details

- [fusion_bench.method.classification.clip_finetune.ImageClassificationFineTuningForCLIP][]
- [fusion_bench.method.classification.continual_clip_finetune.ContinualImageClassificationFineTuningForCLIP][]
- [fusion_bench.method.classification.image_classification_finetune.ImageClassificationFineTuning][]
- [fusion_bench.method.classification.image_classification_finetune.ImageClassificationFineTuning_Test][]

[^1]: (ICLR 2024) Parameter Efficient Multi-task Model Fusion with Partial Linearization. http://arxiv.org/abs/2310.04742
