# MoE-based Model Merging

<figure markdown="span">
    ![alt text](images/sparse_upcycling.png){width="900"}
</figure>

MoE-based model merging is a technique that combines multiple fine-tuned dense models into a single Mixture-of-Experts (MoE) model. This approach leverages the specialization of different expert models by treating each fine-tuned model as an expert in the resulting MoE architecture. The method involves upscaling the architecture to MoE format and substituting the experts with the weights from different specialized models.

## Examples

### API Usage

#### Basic Example

Here's an example demonstrating how to merge multiple fine-tuned models into a Mixtral MoE model:

```python
from fusion_bench.method import (
    MixtralForCausalLMMergingAlgorithm,
    MixtralMoEMergingAlgorithm,
)
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.utils import print_parameters

# Create a model pool with your fine-tuned expert models
model_pool = CausalLMPool(
    models={
        "_pretrained_": "path_to_base_model",
        "expert_1": "path_to_finetuned_model_1",
        "expert_2": "path_to_finetuned_model_2",
        "expert_3": "path_to_finetuned_model_3",
        "expert_4": "path_to_finetuned_model_4",
    },
    tokenizer="path_to_base_model",
    model_kwargs={"torch_dtype": "bfloat16"},
)


# Initialize the merging algorithm with direct parameters
merging_algorithm = MixtralForCausalLMMergingAlgorithm(
    experts_per_token=2,  # Number of experts to activate per token
    save_checkpoint=None  # Optional: path to save the merged model
)

# Run the merging process to get a MoE model
moe_model = merging_algorithm.run(model_pool)

print("Merged MoE model:")
print_parameters(moe_model)

# Save the merged MoE model
moe_model.save_pretrained("path_to_save_moe_model")
```

### CLI Usage

This section provides a guide on how to use the `fusion_bench` command-line interface to merge models using MoE-based merging.

#### Configuration Files

Configuration template for the MoE merging method:

```yaml title="config/method/mixtral_moe_merging.yaml"
--8<-- "config/method/mixtral_moe_merging.yaml"
```

Configuration template for the model pool:

```yaml title="config/modelpool/CausalLMPool/mixtral_moe_merging.yaml"
--8<-- "config/modelpool/CausalLMPool/mixtral_moe_merging.yaml"
```

#### Running MoE Merging

Run the fusion_bench command with MoE merging configuration:

```bash
fusion_bench \
    method=mixtral_moe_merging \
    modelpool=CausalLMPool/mixtral_moe_merging \
    taskpool=dummy # this evaluates parameter counts of the merged model
```

## Implementation Details

- [fusion_bench.method.MixtralMoEMergingAlgorithm][]
- [fusion_bench.method.MixtralForCausalLMMergingAlgorithm][]
