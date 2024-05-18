# Ensemble

Ensemble methods are simple and effective ways to improve the performance of machine learning models. 
They combine the outputs of multiple models to create a stronger model. 


## Examples

create a simple ensemble of CLIP-ViT models for image classification

```bash
fusion_bench method=simple_ensemble \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8 
```

