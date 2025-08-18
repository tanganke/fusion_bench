---
title: Sparse Upcycling
---
# MoE-based Model Upscaling (Sparse Upcycling)

<figure markdown="span">
    ![alt text](images/sparse_upcycling.png){width="900"}
</figure>

Sparse upcycling is a technique used to initialize a sparsely activated Mixture-of-Experts (MoE) model from a dense checkpoint. This approach leverages previously incurred training costs to improve the performance of large models while reducing the computational expense. In the process, dense Transformer blocks are partially replaced with MoE blocks, where the MLPs in a Transformer block are replaced by multiple experts. The experts are chosen based on routing probabilities determined by a router. The initialized MoE model is then further trained to recover the performance. This method results in improved performance for both language and vision models while using only a fraction of the original dense pretraining cost [^1].

## Examples

### Basic Example

Here's an example demonstrating how to upscale a pre-trained Mistral model to a Mixtral model:

```python
import os
from transformers import MistralForCausalLM
from fusion_bench.method import (
    MixtralForCausalLMUpscalingAlgorithm,
)
from fusion_bench.utils import print_parameters

# Load a pre-trained Mistral model
pretrained_model = MistralForCausalLM.from_pretrained(
    "path_to_mistral_model"  # Replace with actual model path
)
print("Pretrained model:")
print_parameters(pretrained_model)
# Output:
# Pretrained model:
# trainable params: 7.24B || all params: 7.24B || trainable%: 100.0000

# Initialize the upscaling algorithm with direct parameters
upscaling_algorithm = MixtralForCausalLMUpscalingAlgorithm(
    num_experts=4,  # Number of expert channels
    experts_per_token=2,  # Experts to choose per token
    save_checkpoint=None  # Optional: path to save the model
)

# Run the upscaling process to get a Mixtral model
mixtral_model = upscaling_algorithm.run(pretrained_model)

print("Mixtral model:")
print_parameters(mixtral_model)
# Output:
# Mixtral model:
# trainable params: 24.15B || all params: 24.15B || trainable%: 100.0000

# Save the upscaled Mixtral model
mixtral_model.save_pretrained("path_to_save_mixtral_model")
```

### API Usage

#### Direct Model Upscaling

```python
from transformers import MistralForCausalLM
from fusion_bench.method.mixture_of_experts.mixtral_upcycling import (
    MixtralForCausalLMUpscalingAlgorithm,
    MixtralUpscalingAlgorithm,
)

# Load source model
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# For CausalLM models (includes lm_head)
causal_lm_algorithm = MixtralForCausalLMUpscalingAlgorithm(
    num_experts=8,
    experts_per_token=2,
    save_checkpoint="./mixtral-8x7b"
)
mixtral_causal_lm = causal_lm_algorithm.run(model)
```

#### Using ModelPool

```python
from fusion_bench import BaseModelPool

# Create a model pool
model_dict = {"_pretrained_": model}
modelpool = BaseModelPool(model_dict)

# Run upscaling with modelpool
mixtral_model = upscaling_algorithm.run(modelpool)
```

A Jupyter notebook example is also available at [our repo](https://github.com/tanganke/fusion_bench/blob/main/examples/moe_based_upscaling.ipynb).

### CLI Usage

This section provides a guide on how to use the `fusion_bench` command-line interface to upscale a Mistral model to a Mixtral model.

#### Configuration Files

Configuration template for the MoE upscaling method:

```yaml title="config/method/mixtral_moe_upscaling.yaml"
--8<-- "config/method/mixtral_moe_upscaling.yaml"
```

Configuration template for the model pool:

```yaml title="config/modelpool/CausalLMPool/mistral-7b.yaml"
--8<-- "config/modelpool/CausalLMPool/mistral-7b.yaml"
```

#### CLI Commands

```bash
fusion_bench \
    method=mixtral_moe_upscaling \
    modelpool=CausalLMPool/mistral-7b \
        modelpool.models._pretrained_=path_to_your_pretrained_model \
    taskpool=dummy # this is a dummy taskpool that does nothing but print the parameter counts of the upscaled model
```

## Implementation Details

- [fusion_bench.method.MixtralUpscalingAlgorithm][]
- [fusion_bench.method.MixtralForCausalLMUpscalingAlgorithm][]

[^1]: Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints. http://arxiv.org/abs/2212.05055
