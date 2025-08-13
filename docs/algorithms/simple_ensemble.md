# Simple Ensemble

Ensemble methods are simple and effective ways to improve the performance of machine learning models. 
They combine the outputs of multiple models to create a stronger model. 


## Examples

```python
from fusion_bench.method import EnsembleAlgorithm

# Instantiate the EnsembleAlgorithm
algorithm = EnsembleAlgorithm()

# Assume we have a list of PyTorch models (nn.Module instances) that we want to ensemble.
models = [...]

# Run the algorithm on the models.
merged_model = algorithm.run(models)
```

## Code Integration

Configuration template for the ensemble algorithm:

```yaml title="config/method/simple_ensemble.yaml"
name: simple_ensemble
```

create a simple ensemble of CLIP-ViT models for image classification tasks.

```bash
fusion_bench \
  method=ensemble/simple_ensemble \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 
```

## Implementation Details

- [fusion_bench.method.SimpleEnsembleAlgorithm][]
