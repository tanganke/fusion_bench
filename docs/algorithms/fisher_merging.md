# (Diagonal) Fisher Merging

The Fisher merging algorithm [^1] is a per-parameter weighed averaging method that assigns weights to the models based on the Fisher information matrix of the models on some labeled data.
The Fisher information matrix $F_\theta$ of a model with parameters $\theta$ can be expressed as:

$$ F_\theta = \mathbb{E}_{x \sim p(x)} \left[ \nabla_\theta \log p(y|x, \theta) \nabla_\theta \log p(y|x, \theta)^T \right] $$

where $p(x)$ is the data distribution, $p(y|x, \theta)$ is the model's output distribution, for example, the softmax output of a classification model, and $\nabla_\theta$ is the gradient with respect to the model's parameters $\theta$.
The Fisher information matrix can be used to estimate the importance of each parameter in the model and thus assign weights to the models based on their Fisher information. 
In addition, the Fisher information matrix can be used to estimate the similarity between tasks, which can be useful in auxiliary-task learning and multi-task learning scenarios [^2].

As the full Fisher information matrix is often computationally expensive to compute and memory-intensive to store, we approximate using the diagonal Fisher information matrix, which is the diagonal of the full Fisher information matrix.
The diagonal Fisher information matrix can be computed as:

$$ \hat{F}_\theta = \mathbb{E}_{x \sim p(x)} \left[ \left(\nabla_\theta \log p(y|x, \theta)\right)^2 \right] $$

Assuming we have $n$ models with parameters $\theta_i$ and diagonal Fisher information matrices $\hat{F}_{\theta_i}$, the Fisher merging algorithm computes the merged model's parameters $\theta$ as follows:

$$ \theta^{(j)} = \frac{\sum_{i=1}^{n} \hat{F}_{\theta_i}^{(j)} \theta_i^{(j)}}{\sum_{i=1}^{n} \hat{F}_{\theta_i}^{(j)}} $$

where $\theta_i$ are the parameters of the individual models, $\hat{F}_{\theta_i}$ are the diagonal Fisher information matrices of the individual models, and $j$ indexes the parameters of the models.
The Fisher merging algorithm can be considered a per-weight weighed averaging method, where the weights are determined by the Fisher information of each parameter in the models.

## Code Integration

Example of merging eight CLIP-ViT-B/32 models using Fisher merging:

```bash
fusion_bench method=clip_fisher_merging \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8
```

Merge eight CLIP-ViT-L/14 models using Fisher merging:

```bash
fusion_bench \
  method=clip_fisher_merging \
    method.batch_size=8 method.num_workers=4 \
  modelpool=clip-vit-large-patch14_TA8 \
  taskpool=clip-vit-classification_TA8 \
    taskpool.clip_model=openai/clip-vit-large-patch14
```

Merge GPT-2 models for text classification tasks:

```bash
fusion_bench \
  method=gpt2_fisher_merging \
    method.num_fisher_examples=512 method.batch_size=8 \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

## References

::: fusion_bench.method.fisher_merging.fisher_merging.FisherMergingAlgorithm

[^1]: M. Matena, C. Raffel. "Merging Models with Fisher-Weighted Averaging" http://arxiv.org/abs/2111.09832
[^2]: C. Wu, et al. "Pi-Tuning: Transferring Multimodal Foundation Models with Optimal Multi-task Interpolation". https://github.com/TencentARC/pi-Tuning

