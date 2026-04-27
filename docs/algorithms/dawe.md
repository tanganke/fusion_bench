# DAWE (Data-Adaptive Weight Ensembling)

DAWE is a data-adaptive model ensembling method that learns a gating mechanism at inference time to dynamically route inputs through different expert models. Unlike static merging approaches that compute a fixed set of weights, DAWE uses a learned neural network gate that conditions on both the input data and task-specific feature embeddings to produce soft routing weights. This enables the merged model to selectively leverage the strengths of different expert models for different inputs.

## Algorithm Overview

DAWE addresses a fundamental limitation of static model merging: a single fixed set of merge weights cannot optimally combine models for all possible inputs. Instead, DAWE learns a data-dependent routing function that produces different ensembling weights for each input sample.

### Architecture

The DAWE system consists of three components:

1. **Expert Models**: A set of fine-tuned models $\{\theta_1, \theta_2, ..., \theta_N\}$, each specialized for a particular task. A pretrained base model $\theta_0$ serves as the reference point.

2. **Feature Extractor**: A separate model (e.g., ResNet-18) that extracts task-discriminative features from the input. This feature extractor processes the raw input and provides a representation that captures which task the input belongs to.

3. **Gating Network**: A small neural network that takes the feature extractor's output as input and produces routing weights for the expert models. The gate has configurable hidden layers (`gate_hidden_layers`) and its parameters are learned during test-time adaptation.

### Inference Process

At inference time, for an input image $x$:

1. The CLIP vision model extracts visual features via its `pooler_output`.
2. The ResNet-based feature extractor processes the raw image to obtain task-discriminative features.
3. The gating network maps the feature extractor output to routing weights $w \in \mathbb{R}^{N+1}$ (including the base model).
4. The final output is a weighted combination of the expert models' outputs.

### Merge Modes

DAWE supports two merging granularity modes:

- **`task_wise`**: A single routing weight per model (model-level mixing).
- **`layer_wise`**: Per-layer routing weights (layer-level mixing).

### Batch Reduction

The `batch_reduce` option enables reducing the routed outputs within a batch, which can be useful for generating batch-level aggregated predictions.

## Mathematical Formulation

### Task Vector Representation

Each expert model $i$ is represented as a task vector relative to the pretrained model:

$$\tau_i = \theta_i - \theta_0$$

### Gating Network

The gating network $g_\phi$ is parameterized by learnable parameters $\phi$. Given input features $f$ from the feature extractor:

$$w = \text{softmax}(g_\phi(f))$$

where $w \in \mathbb{R}^{N+1}$ are the routing weights, and the softmax ensures they sum to 1.

### Weighted Ensemble

The final merged representation is:

$$\theta_{\text{merged}}(x) = \sum_{i=0}^{N} w_i(x) \cdot \tau_i$$

where $w_i(x) = g_\phi(f(x))_i$ are the input-dependent weights.

The base model output is added back:

$$\text{output}(x) = \theta_0 + \theta_{\text{merged}}(x)$$

### Task Vector Sparsity

For efficiency, task vectors can be sparsified by keeping only the top-$k$ most important parameters:

$$\tau_i \leftarrow \tau_i \cdot \mathbb{1}_{\{|\tau_i| \geq \text{threshold}\}}$$

controlled by the `task_vector_sparsity` parameter.

### Training Objective

The gate parameters $\phi$ are optimized via entropy minimization on the model's predictions:

$$\mathcal{L}_{\text{entropy}} = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{c} p(c|x; \theta_{\text{merged}}) \log p(c|x; \theta_{\text{merged}}) \right]$$

The gradient flows through the gate, the expert outputs, and the routing weights, enabling end-to-end optimization.

## Configuration

```yaml title="config/method/dawe/dawe_for_clip.yaml"
--8<-- "config/method/dawe/dawe_for_clip.yaml"
```

Key configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `merge_mode` | Merging granularity: `task_wise` or `layer_wise` | `task_wise` |
| `init_lambda` | Initial merge weight for the gate | `0.3` |
| `batch_reduce` | Whether to reduce within batch | `true` |
| `dict_feature_extractor_path` | Path to the feature extractor model | `microsoft/resnet-18` |
| `hidden_size` | Dimension of extracted features (inferred if null) | `null` |
| `gate_hidden_layers` | Number of hidden layers in the gate | `1` |
| `task_vector_sparsity` | Sparsity ratio for task vectors | `0` |
| `max_steps` | Number of training steps for the gate | `1000` |
| `learning_rate` | Learning rate for gate optimization | `1e-5` |
| `skip_training` | Skip gate training (use initial weights) | `false` |

## Examples

### CLI Usage

```bash
fusion_bench \
    method=dawe/dawe_for_clip \
    method.merge_mode=task_wise \
    method.max_steps=1000 \
    method.learning_rate=1e-5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method.dawe.dawe_for_clip import DataAdaptiveWeightEnsemblingForCLIP
from fusion_bench.modelpool import CLIPVisionModelPool

# Create the algorithm
algorithm = DataAdaptiveWeightEnsemblingForCLIP(
    merge_mode="task_wise",
    init_lambda=0.3,
    batch_reduce=True,
    max_steps=1000,
    learning_rate=1e-5,
)

# Run on a model pool
modelpool = CLIPVisionModelPool(...)
merged_model = algorithm.run(modelpool)
```

## Implementation Details

- **`DataAdaptiveWeightEnsemblingCLIPVisionModel`**: The core wrapper model that combines the CLIP vision model, feature extractor, and gating network. Forward pass routes inputs through experts based on gate predictions.
- **`ResNetFeatureExtractor`**: A wrapper around `ResNetForImageClassification` that removes the classification head and flattens to produce feature vectors.
- **`load_resnet_processor`**: Loads a ResNet processor for image preprocessing, handling RGB conversion.
- **Checkpoints**: During training, checkpoints are saved at every `save_interval` steps to `log_dir/checkpoints/model_{step}.pt`.

## References

[^1]: (ICLR 2024) DAWE: Data-Adaptive Weight Ensembling for Pre-Trained Models. http://arxiv.org/abs/2310.02575. Introduces the data-adaptive ensembling framework with learnable routing.
