# Ensemble

Ensemble methods are simple and effective ways to improve the performance of machine learning models. 
They combine the outputs of multiple models to create a stronger model. 


## Examples

```python
from fusion_bench.method.

## Code Integration

### Simple Ensemble

Configuration template for the ensemble algorithm:

```yaml title="config/method/simple_ensemble.yaml"
name: simple_ensemble
```

create a simple ensemble of CLIP-ViT models for image classification

```bash
fusion_bench method=simple_ensemble \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8 
```

### Weighted Ensemble

Configuration template for the weighted ensemble algorithm:

```yaml title="config/method.weighted_ensemble.yaml"
name: weighted_ensemble

# this should be a list of floats, one for each model in the ensemble
# If weights is null, the ensemble will use the default weights, which are equal weights for all models.
weights: null
```

or create a weighted ensemble of CLIP-ViT models for image classification

```bash
fusion_bench method=weighted_ensemble \
    method.weights=[0.5, 0.5] \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8
```
