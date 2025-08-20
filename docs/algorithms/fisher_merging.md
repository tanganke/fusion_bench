---
title: Fisher Merging
---
# (Diagonal) Fisher Merging

Fisher merging [^1] is a parameter-weighted averaging method that assigns weights to model parameters based on the Fisher information matrix computed on labeled data.
This approach allows for more informed model combination by considering the importance of each parameter as indicated by the Fisher information.

## Mathematical Foundation

The Fisher information matrix $F_\theta$ of a model with parameters $\theta$ is defined as:

$$ F_\theta = \mathbb{E}_{x \sim p(x)} \left[ \nabla_\theta \log p(y|x, \theta) \nabla_\theta \log p(y|x, \theta)^T \right] $$

where:

- $p(x)$ is the data distribution
- $p(y|x, \theta)$ is the model's output distribution (e.g., softmax output for classification)
- $\nabla_\theta$ is the gradient with respect to the model's parameters $\theta$

The Fisher information matrix quantifies the importance of each parameter and can estimate task similarity, making it valuable for auxiliary-task learning and multi-task learning scenarios [^2].

## Diagonal Fisher Approximation

Since the full Fisher information matrix is computationally expensive and memory-intensive, we use the diagonal Fisher information matrix approximation:

$$ \hat{F}_\theta = \mathbb{E}_{x \sim p(x)} \left[ \left(\nabla_\theta \log p(y|x, \theta)\right)^2 \right] $$

Given $n$ models with parameters $\theta_i$ and diagonal Fisher information matrices $\hat{F}_{\theta_i}$, the Fisher merging algorithm computes the merged model's parameters as:

$$ \theta^{(j)} = \frac{\sum_{i=1}^{n} \hat{F}_{\theta_i}^{(j)} \theta_i^{(j)}}{\sum_{i=1}^{n} \hat{F}_{\theta_i}^{(j)}} $$

where $j$ indexes individual parameters. This creates a per-parameter weighted average where weights are determined by the Fisher information of each parameter.

## Examples

### CLI Usage

#### CLIP Vision Model Fisher Merging

Configuration template for CLIP Fisher merging:

```yaml title="config/method/fisher_merging/clip_fisher_merging.yaml"
--8<-- "config/method/fisher_merging/clip_fisher_merging.yaml"
```

Example merging eight CLIP-ViT-B/32 models:

```bash
fusion_bench method=fisher_merging/clip_fisher_merging \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Merge eight CLIP-ViT-L/14 models with custom batch settings:

```bash
fusion_bench \
  method=fisher_merging/clip_fisher_merging \
    method.dataloader_kwargs.batch_size=8 \
    method.dataloader_kwargs.num_workers=4 \
  modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    taskpool.clip_model=openai/clip-vit-large-patch14
```

#### GPT-2 Fisher Merging

Configuration template for GPT-2 Fisher merging:

```yaml title="config/method/fisher_merging/gpt2_fisher_merging.yaml"
--8<-- "config/method/fisher_merging/gpt2_fisher_merging.yaml"
```

Example merging GPT-2 models for text classification:

```bash
fusion_bench \
  method=fisher_merging/gpt2_fisher_merging \
    method.num_fisher_examples=512 \
    method.batch_size=8 \
    method.num_workers=2 \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

### API Usage

#### CLIP Fisher Merging

```python
from fusion_bench.method.fisher_merging.clip_fisher_merging import FisherMergingForCLIPVisionModel

algorithm = FisherMergingForCLIPVisionModel(
    exclude_param_names_regex=[],
    normalize_fisher_weight=True,
    minimal_fisher_weight=1e-6,
    num_fisher_examples=256,
    dataloader_kwargs={
        "batch_size": 32,
        "num_workers": 4
    },
)

merged_model = algorithm.run(modelpool)
```

#### GPT-2 Fisher Merging

```python
from fusion_bench.method.fisher_merging.gpt2_fisher_merging import FisherMergingAlgorithmForGPT2

algorithm = FisherMergingAlgorithmForGPT2(
    exclude_param_names_regex=[],
    normalize_fisher_weight=True,
    minimal_fisher_weight=1e-6,
    num_fisher_examples=256,
    cache_dir="outputs",
    batch_size=32,
    num_workers=0
)

merged_model = algorithm.run(modelpool)
```

## Implementation Details

- [`fusion_bench.method.fisher_merging.FisherMergingAlgorithm`][fusion_bench.method.FisherMergingAlgorithm]: Base Fisher merging implementation
- [`fusion_bench.method.fisher_merging.clip_fisher_merging.FisherMergingForCLIPVisionModel`][fusion_bench.method.FisherMergingForCLIPVisionModel]: CLIP vision model specialization
- [`fusion_bench.method.fisher_merging.gpt2_fisher_merging.FisherMergingAlgorithmForGPT2`][fusion_bench.method.FisherMergingAlgorithmForGPT2]: GPT-2 text classification specialization

[^1]: M. Matena, C. Raffel. "Merging Models with Fisher-Weighted Averaging" http://arxiv.org/abs/2111.09832
[^2]: C. Wu, et al. "Pi-Tuning: Transferring Multimodal Foundation Models with Optimal Multi-task Interpolation". https://github.com/TencentARC/pi-Tuning

