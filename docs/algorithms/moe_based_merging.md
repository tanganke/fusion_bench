# MoE-based Model Model Merging

## Code Intergration

Here we provides instructions on how to use the `fusion_bench` command-line interface to merge models using a Mixture of Experts (MoE) approach.

The first code block is a YAML configuration file for the merging method. The `name` field specifies the name of the merging method. The `num_experts` field specifies the number of experts to use in the merging process. The `experts_per_token` field specifies the number of experts to use per token. The `save_checkpoint` field specifies the path where the merged model will be saved.

```yaml title="config/method/mixtral_moe_merging.yaml"
name: mixtral_for_causal_lm_moe_merging

experts_per_token: 2
# path to save the merged model, if provided
save_checkpoint: null
```

The second code block is another YAML configuration file, this time for the model pool. The `type` field specifies the type of model pool to use. The `models` field is a list of models to include in the pool. Each model should have a `name` and a `path`, and the model is loaded from the path.

```yaml title="config/modelpool/mixtral_moe_merging.yaml"
type: AutoModelForCausalLMPool
# each model should have a name and a path, and the model is loaded from the path
# this is equivalent to `AutoModelForCausalLM.from_pretrained(path)`
models:
  - name: _pretrained_
    path: path_to_your_pretrained_model
  - name: expert_1
    path: path_to_your_expert_model_1
  - name: expert_2
    path: path_to_your_expert_model_2
  - name: expert_3
    path: path_to_your_expert_model_3
  - name: expert_4
    path: path_to_your_expert_model_4
```

Finally, the third code block is a bash command that runs the `fusion_bench` command-line interface with the specified method, model pool, and task pool. The `method` argument specifies the merging method to use. The `modelpool` argument specifies the model pool to use. The `modelpool.models.0.path` argument specifies the path to the pretrained model to use. The `taskpool` argument specifies the task pool to use. In this case, a dummy task pool is used that does nothing but print the parameter counts of the merged model.

```bash
fusion_bench \
    method=mixtral_moe_merging \
    modelpool=mixtral_moe_merging \
    taskpool=dummy # this is a dummy taskpool that does nothing but print the parameter counts of the merged model
```

This guide provides a step-by-step process for merging models using the `fusion_bench` command-line interface. By following these instructions, you can merge your own models and save them for future use.

## References

::: fusion_bench.method.mixture_of_experts.mixtral_merging

