# Concrete Subspace Learning

## Code Integration

Merging CLIP models on eight image classification tasks, using the concrete task arithmetic algorithm

```bash
# tensorboard logs and learned checkpoints of the shared mask can be found at https://huggingface.co/tanganke/clip-vit-base-patch32_concrete-task-arithmetic_tblogs
fusion_bench \
    fabric_logger.name=ViT-B-32/concrete_task_arithmetic \
    method=clip_concrete_task_arithmetic \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8
```

results

```json
{
    "svhn": {
        "accuracy": 0.903003990650177,
        "loss": 0.37700024247169495
    },
    "stanford_cars": {
        "accuracy": 0.6326327323913574,
        "loss": 1.2553859949111938
    },
    "resisc45": {
        "accuracy": 0.7558730244636536,
        "loss": 1.017554759979248
    },
    "eurosat": {
        "accuracy": 0.9407407641410828,
        "loss": 0.20871955156326294
    },
    "gtsrb": {
        "accuracy": 0.8285035490989685,
        "loss": 0.5861473679542542
    },
    "mnist": {
        "accuracy": 0.9800000190734863,
        "loss": 0.08148527890443802
    },
    "dtd": {
        "accuracy": 0.5249999761581421,
        "loss": 2.2731478214263916
    },
    "sun397": {
        "accuracy": 0.6421158909797668,
        "loss": 1.4108904600143433
    }
}
```

Concrete AdaMerging (Layer-wise)

```bash
# tensorboard logs and learned checkpoints of the shared mask can be found at https://huggingface.co/tanganke/clip-vit-base-patch32_concrete-layer-wise_adamerging_tblogs
fusion_bench \
    fabric_logger.name=ViT-B-32/clip_concrete_layer_wise_adamerging \
    method=clip_concrete_layer_wise_adamerging \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8
```

## Further Reading

- :llama: [:simple-github:](https://github.com/xinykou/safety_realignment) 
    X. Yi, S. Zheng, L. Wang, X. Wang, and L. He, “A safety realignment framework via subspace-oriented model fusion for large language models.” [arXiv, May 14, 2024. doi: 10.48550/arXiv.2405.09055.](http://arxiv.org/abs/2405.09055)

    > The paper introduces a safety realignment framework for large language models via subspace-oriented model fusion (SOMF, the authors learn a shared mask on the weight space of large language model), which combines safeguard capabilities of initially aligned models with fine-tuned models to ensure safety without compromising performance on downstream tasks.
