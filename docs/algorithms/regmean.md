# RegMean

[![arXiv](https://img.shields.io/badge/arXiv-2212.09849-b31b1b.svg)](http://arxiv.org/abs/2212.09849)

RegMean (Regression Mean) is a dataless knowledge fusion approach that formulates model merging as a linear regression problem[^1]. The algorithm aims to find optimal weights for each linear layer in the merged model by minimizing the discrepancy in predictions between the merge and candidate models.

## Algorithm Overview

For a transformer layer $l$, to obtain the merge weights for a linear layer $W^{(l)}_{M}$, RegMean provides a precise closed-form solution for merging those from $K$ candidate models:

$$W^{(l)}_{M} = \left[\sum_{i=1}^{K}  (X^{(l)}_i)^{\top} X^{(l)}_i\right]^{-1} \sum_{i=1}^{K} (X^{(l)}_i)^{\top} X^{(l)}_i W^{(l)}_i$$

where:

- $W^{(l)}_i$ is the weight matrix of the $i$-th candidate model at layer $l$
- $X^{(l)}_i$ represents the input activations to layer $l$ for model $i$
- The formula computes a weighted combination that minimizes prediction discrepancy

## Examples

### CLI Usage

Configuration templates for RegMean:

```yaml title="config/method/regmean/clip_regmean.yaml"
--8<-- "config/method/regmean/clip_regmean.yaml"
```

```yaml title="config/method/regmean/gpt2_regmean.yaml"
--8<-- "config/method/regmean/gpt2_regmean.yaml"
```

#### CLIP Models

Merge CLIP-ViT-B/32 models on eight image classification tasks:

```bash
fusion_bench method=regmean/clip_regmean \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Merge CLIP-ViT-L/14 models on eight image classification tasks:

```bash
fusion_bench \
  method=regmean/clip_regmean \
    method.dataloader_kwargs.batch_size=8 \
  modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    taskpool.base_model=openai/clip-vit-large-patch14
```

#### Language Models

Merge GPT-2 models for text classification tasks:

```bash
fusion_bench \
  method=regmean/gpt2_regmean \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

### API Usage

To use RegMean programmatically:

#### CLIP Models

```python
from fusion_bench.method.regmean import RegMeanAlgorithmForCLIP
from omegaconf import DictConfig

# Configuration for CLIP RegMean
config = DictConfig({
    'exclude_param_names_regex': [],
    'num_regmean_examples': 256,
    'weight_transpose': True,
    'reduce_non_diagonal_ratio': 0.95,
    'dataloader_kwargs': {
        'batch_size': 32,
        'num_workers': 0
    }
})

# Initialize the algorithm
algorithm = RegMeanAlgorithmForCLIP(**config)

# Run the algorithm with a model pool
merged_model = algorithm.run(modelpool)
```

#### GPT-2 Models

```python
from fusion_bench.method.regmean import RegMeanAlgorithmForGPT2
from omegaconf import DictConfig

# Configuration for GPT-2 RegMean
config = DictConfig({
    'exclude_param_names_regex': [],
    'num_regmean_examples': 256,
    'reduce_non_diagonal_ratio': 0.6,
    'weight_transpose': False,
    'cache_dir': 'outputs',
    'batch_size': 32,
    'num_workers': 0
})

# Initialize the algorithm
algorithm = RegMeanAlgorithmForGPT2(**config)

# Run the algorithm
merged_model = algorithm.run(modelpool)
```

## Implementation Details

- [fusion_bench.method.RegMeanAlgorithmForCLIP][]
- [fusion_bench.method.RegMeanAlgorithmForGPT2][]


[^1]: Xisen Jin, et al. "Dataless Knowledge Fusion by Merging Weights of Language Models." http://arxiv.org/abs/2212.09849
