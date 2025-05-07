# Isotropic Merging

[![arXiv](https://img.shields.io/badge/arXiv-2502.04959-b31b1b.svg?style=flat)](https://arxiv.org/abs/2502.04959)

![alt text](images/iso_merging.png){ width="750"}

## Code Integration

### CLIP-ViT-B/32 on 8 tasks

Merge CLIP-ViT-B/32 models on eight image classification tasks using ISO-C, with a scaling factor of 1.5:

```bash
fusion_bench \
    fabric.loggers.name=iso_c \
    method=isotropic_merging/iso_c \
    method.scaling_factor=1.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Merge CLIP-ViT-B/32 models on eight image classification tasks using ISO-CTS, with a scaling factor of 1.5:

```bash
fusion_bench \
    fabric.loggers.name=iso_cts \
    method=isotropic_merging/iso_cts \
    method.scaling_factor=1.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### CLIP-ViT-L/14 on eight tasks

Merge CLIP-ViT-L/14 models on eight image classification tasks using ISO-C, with a scaling factor of 1.5:

```bash
fusion_bench \
    fabric.loggers.name=iso_c \
    method=isotropic_merging/iso_c \
    method.scaling_factor=1.5 \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14
```

