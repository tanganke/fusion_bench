# Fine-tuning ResNet Models for Image Classification

This guide demonstrates how to fine-tune ResNet models for image classification tasks using FusionBench. The framework supports various ResNet architectures on different datasets.

## Quick Start

### Training a ResNet-18 Model on CIFAR-10

```bash
fusion_bench --config-name model_fusion \
    path.log_dir=outputs/resnet18/cifar10 \
    method=classification/image_classification_finetune \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10
```

This command will start fine-tuning a ResNet-18 model on the CIFAR-10 dataset using all available GPUs. 
The global batch size is determined by the number of GPUs multiplied by the batch size per GPU (specified as `method.dataloader_kwargs.batch_size`).
The training outputs, including checkpoints and logs, will be saved in the specified directory (in this example, `outputs/resnet18/cifar10`).


### Testing the Fine-tuned Model

After training, test your model using the saved checkpoint:

```bash
fusion_bench --config-name model_fusion \
    method=classification/image_classification_finetune_test \
    method.checkpoint_path=<path_to_your_checkpoint> \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10
```

**Example with actual checkpoint path:**

```bash
fusion_bench --config-name model_fusion \
    method=classification/image_classification_finetune_test \
    method.checkpoint_path="outputs/resnet18/cifar10/version_0/checkpoints/epoch\=9-step\=1960.ckpt" \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10
```

## Training Configuration

### Default Training Parameters

The default training configuration (`config/method/classification/image_classification_finetune.yaml`) includes:

```yaml title="config/method/classification/image_classification_finetune.yaml"
--8<-- "config/method/classification/image_classification_finetune.yaml"
```

### Customizing Training Parameters

You can override any training parameter from the command line:

```bash
# Custom learning rate and batch size
fusion_bench --config-name model_fusion \
    method=classification/image_classification_finetune \
    method.optimizer.lr=0.01 \
    method.dataloader_kwargs.batch_size=128 \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10

# Different optimizer (Adam)
fusion_bench --config-name model_fusion \
    method=classification/image_classification_finetune \
    method.optimizer._target_=torch.optim.Adam \
    method.optimizer.lr=0.001 \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10

# Step-based training instead of epoch-based
fusion_bench --config-name model_fusion \
    method=classification/image_classification_finetune \
    method.max_epochs=null \
    method.max_steps=5000 \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10
```

## Testing Configuration

The testing configuration (`config/method/classification/image_classification_finetune_test.yaml`) includes:

```yaml title="config/method/classification/image_classification_finetune_test.yaml"
--8<-- "config/method/classification/image_classification_finetune_test.yaml"
```

### Testing Options

```bash
# Test with checkpoint
fusion_bench --config-name model_fusion \
    method=classification/image_classification_finetune_test \
    method.checkpoint_path="path/to/checkpoint.ckpt" \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10

# Test without checkpoint (using pretrained weights)
fusion_bench --config-name model_fusion \
    method=classification/image_classification_finetune_test \
    modelpool=ResNetForImageClassfication/transformers/resnet18_cifar10
```

### Monitoring Training

View training progress using TensorBoard:

```bash
cd outputs
tensorboard --logdir .
```

The following metrics are logged:
- Training and validation loss
- Top-1 and Top-5 accuracy
- Learning rate schedule
- Device statistics (GPU utilization, memory usage)
