# RegMean

## Code Integration

Merge CLIP-ViT-B/32 models on eight image classification tasks

```bash
fusion_bench method=clip_regmean \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8
```

Merge CLIP-ViT-L/14 models on eight image classification tasks

```bash
fusion_bench \
  method=clip_regmean \
    method.batch_size=8 method.num_workers=4 \
  modelpool=clip-vit-large-patch14_TA8 \
  taskpool=clip-vit-classification_TA8 \
    taskpool.clip_model=openai/clip-vit-large-patch14
```

Merge GPT-2 models for text classification tasks:

```bash
fusion_bench \
  method=gpt2_regmean \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

## References

[^1]: Xisen Jin, et al. "Dataless Knowledge Fusion by Merging Weights of Language Models." http://arxiv.org/abs/2212.09849
