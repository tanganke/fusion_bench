# Gossip

Gossip merging is a decentralized model fusion strategy that simulates distributed model merging through iterative pairwise communication. Instead of performing a single global merge, models exchange knowledge with their neighbors over multiple rounds, each round performing local adaptive merging. This approach better simulates real-world scenarios where models are distributed across different nodes and cannot access a central model pool simultaneously.

The Gossip algorithm builds upon the AdaMerging framework, combining two key ideas:

1. **Layer-wise or task-wise adaptive merging**: Learn optimal merging weights via test-time adaptation using entropy minimization.
2. **Gossip communication topology**: Models communicate with neighbors in a ring or rotated topology, propagating knowledge incrementally across rounds.

## Algorithm Overview

### Model Scheduler

The `ModelScheduler` class manages model storage and determines which models participate in each gossip round. For each model `i` in a round, the scheduler constructs a local merging environment based on the topology:

- **Ring topology (`ring`)**: Model `i` merges with its left and right neighbors, i.e., models `(i-1) % N`, `i`, and `(i+1) % N`.
- **Rotate topology (`rotate_K`)**: Model `i` merges with itself and the next `K` models in cyclic order: models `i`, `(i+1) % N`, ..., `(i+K) % N`.

### Gossip Merging Process

The algorithm proceeds in `gossip_max_steps` rounds. In each round:

1. For each model `i` from `0` to `N-1`:
   - The scheduler selects the neighbor models according to the topology.
   - A `LayerWiseMergedModel` (or `TaskWiseMergedModel`) is constructed with the pretrained model and the selected fine-tuned models.
   - Test-time adaptation optimizes the layer-wise (or task-wise) merging weights via entropy minimization.
   - The resulting merged weights are stored back as the updated fine-tuned model `i`.
2. After all models have been processed, the scheduler updates the model pool with the new models.

The process iterates, allowing knowledge to propagate through the network. The `improve_dataset` flag controls whether each local merge only uses data from tasks involved in that local merge.

## Mathematical Formulation

### Layer-Wise Merging Weights

For layer-wise gossip, let $W^{(i)} \in \mathbb{R}^{L \times M_i}$ denote the layer-wise merging weight matrix for model $i$, where $L$ is the number of layers and $M_i$ is the number of models involved in model $i$'s local merge. The merged parameter for layer $l$ is computed as:

$$\theta_{\text{merged}}^{(l)} = \sum_{j=1}^{M_i} W^{(i)}_{l,j} \cdot (\theta_j^{(l)} - \theta_0^{(l)}) + \theta_0^{(l)}$$

where $\theta_j^{(l)}$ is the parameter of fine-tuned model $j$ at layer $l$, and $\theta_0^{(l)}$ is the pretrained parameter.

### Task-Wise Merging Weights

For task-wise gossip, a single scalar weight per model is learned:

$$\theta_{\text{merged}} = \sum_{j=1}^{M_i} W^{(i)}_j \cdot (\theta_j - \theta_0) + \theta_0$$

### Entropy Minimization

The merging weights are optimized by minimizing the entropy of the model's predictions on a small calibration set:

$$\mathcal{L}_{\text{entropy}} = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{c} p(c|x) \log p(c|x) \right]$$

where $p(c|x) = \text{softmax}(z_c(x))$ is the softmax probability distribution over classes, and $z_c(x)$ denotes the logit for class $c$ on input $x$. The gradient of this loss flows through the merging weights, enabling differentiable optimization of the merge.

### Weight Constraints

The merging weights can be optionally constrained:

$$W \in [0, 1] \quad \text{(if `clamp_weights` is enabled)}$$

$$\sum_{j} W_j = 1 \quad \text{(if `tie_weights` is enabled)}$$

## Variants

### CLIPLayerWiseGossipAlgorithm

The CLIP-specific variant (`clip_layer_wise_gossip.py`) extends `LayerWiseGossipAlgorithm` with CLIP zero-shot classification capabilities. It sets up a zero-shot classification head using the CLIP text encoder and computes logits via the CLIP vision model's pooled output.

### FlanT5LayerWiseGossipAlgorithm

The Flan-T5 variant (`flan_t5_layer_wise_gossip.py`) adapts the gossip algorithm for the Flan-T5 language model, using sequence classification heads for test-time adaptation.

### TaskWiseGossipAlgorithm

The task-wise variant (`task_wise_gossip.py`) performs merging at the model level rather than the layer level. Each local merge learns a single scalar weight per model, controlling how much of each model's task vector contributes to the merged result.

## Configuration

### Layer-Wise CLIP Gossip

```yaml title="config/method/gossip/layer_wise_clip.yaml"
--8<-- "config/method/gossip/layer_wise_clip.yaml"
```

### Layer-Wise Flan-T5 Gossip

```yaml title="config/method/gossip/layer_wise_flan_t5.yaml"
--8<-- "config/method/gossip/layer_wise_flan_t5.yaml"
```

## Examples

### CLI Usage

Run layer-wise gossip merging on CLIP models:

```bash
fusion_bench \
    method=gossip/layer_wise_clip \
    method.gossip_max_steps=20 \
    method.max_steps=400 \
    method.topo=ring \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Run gossip merging with rotate topology (each model communicates with 2 neighbors):

```bash
fusion_bench \
    method=gossip/layer_wise_clip \
    method.topo=rotate_2 \
    method.gossip_max_steps=15 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Skip the AdaMerging step (only perform static merging with gossip):

```bash
fusion_bench \
    method=gossip/layer_wise_clip \
    method.gossip_skip_adamerging=true \
    method.gossip_max_steps=10 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

## Implementation Details

The implementation uses several key components:

- **ModelScheduler** (`ModelScheduler` class): Manages model loading, neighbor selection, and storage of updated models. The `update_datasets` method adjusts which datasets are used for each local merge when `improve_dataset` is enabled.
- **LayerWiseMergedModel**: A wrapper model that performs differentiable layer-wise interpolation between the pretrained and fine-tuned models.
- **TaskWiseMergedModel**: A wrapper model that performs differentiable task-wise interpolation at the model level.
- **Entropy loss**: The optimization objective for test-time adaptation, encouraging the merged model to make confident predictions.
- **Memory management**: The `free_gpu_memory` method explicitly moves models to CPU and clears GPU cache between merges to simulate distributed memory constraints.

## References

[^1]: (NeurIPS 2023) Communication-Efficient On-Device Machine Learning: Federated Learning and Beyond. Discusses gossip-based distributed optimization for model aggregation.
[^2]: (ICLR 2024) AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575. The base adaptive merging algorithm extended by the gossip framework.
