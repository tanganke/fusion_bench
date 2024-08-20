# Fine-Tune Your Own Vision Transformer

In this guide, we will show you how to fine-tune your own Vision Transformer (ViT) model on a custom dataset using `fusion_bench` CLI. 
FusionBench provides a simple and easy-to-use interface to fine-tune clip vision transformer in a single-task learning setting or traditional multi-task learning setting.

## Basic Examples

### Multi-Task Learning

Fine-tune CLIP-ViT-B/32:

```bash
fusion_bench \
    method=clip_finetune \
    modelpool=clip-vit-base-patch32_mtl \
    taskpool=dummy
```

Fine-tune CLIP-ViT-L/14 on eight GPUs with a per-device per-task batch size of 2.

```bash
fusion_bench \
    fabric.devices=8 \
    method=clip_finetune \
        method.batch_size=2 \
    modelpool=clip-vit-base-patch32_mtl \
        modelpool.models.0.path=openai/clip-vit-large-patch14 \
    taskpool=dummy
```

This will save the state dict of the vision model (`transformers.models.clip.CLIPVisionModel.CLIPVisionTransformer`) to the log directory.
Subsequently, we can use `fusion_bench/scripts/clip/convert_checkpoint.py` to convert the state dict to a HuggingFace model (`CLIPVisionModel`).

=== "method configuration"

    ```yaml title="config/method/clip_finetune.yaml"
    name: clip_finetune

    seed: 42

    learning_rate: 1e-5
    num_steps: 4000

    batch_size: 32
    num_workers: 4

    save_interval: 500
    ```

=== "model pool configuration"

    ```yaml title="config/modelpool/clip-vit-base-patch32_mtl.yaml"
    type: huggingface_clip_vision
    models:
      - name: _pretrained_
        path: openai/clip-vit-base-patch32
    
    dataset_type: huggingface_image_classification
    train_datasets:
      - name: svhn
        dataset:
          type: instantiate
          name: svhn
          object:
            _target_: datasets.load_dataset
            _args_:
              - svhn
              - cropped_digits
            split: train
      - name: stanford_cars
        dataset:
          name: tanganke/stanford_cars
          split: train
      # other datasets
      # ...
    ```

```bash
# or CLIP-ViT-L/14, add option: --model openai/clip-vit-large-patch14
python fusion_bench/scripts/clip/convert_checkpoint.py \
    --checkpoint /path/to/checkpoint \
    --output /path/to/output
```

After converting the checkpoint, you can use FusionBench to evaluate the model.
For example, you can use the following command to evaluate the model on the eight tasks documented [here](../../modelpool/clip_vit.md).

```bash
path_to_clip_model=/path/to/converted/output
fusion_bench method=dummy \
  modelpool=clip-vit-base-patch32_individual \
    modelpool.models.0.path="'${path_to_clip_model}'" \
  taskpool=clip-vit-classification_TA8
```

### Single-Task Learning

Simply remove some of the datasets from the `train_datasets` field in the model pool configuration.


## References

::: fusion_bench.method.classification.clip_finetune.ImageClassificationFineTuningForCLIP

