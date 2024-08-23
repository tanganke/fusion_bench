# Weighted Averaging

Weighted averaging, also known as weight-ensembling.
In the context of full fine-tuned models, the weights are averaged according to their respective performance weights. Concretely, this means that if we have $n$ models with their respective weights $\theta_i$ and model-wise weights $w_i$, the weights of the final model $\theta$ are computed as:

$$ \theta = \sum_{i=1}^{n} w_i \theta_i $$

## Examples

### General Usage

Configuration template for the Weighted Averaging algorithm:

```yaml title="config/method/weighted_average.yaml"
name: weighted_average
normalize: true # if true, the weights will be normalized before merging
weights: # List of weights for each model
  - 0.5
  - 0.5
```

Use the following command to run the Weighted Averaging algorithm:

```bash
fusion_bench method=weighted_average ...
```

### Merge Llama/Mistral Models

Here is an example of how to use the Weighted Averaging algorithm to merge two LLama models. In particular, LLaMa models of the type `transformers.LlamaForCausalLM` are merged using the Weighted Averaging algorithm.

```bash
fusion_bench \
    method=weighted_average_for_llama \
    method.merged_model_save_path=outputs/test_merged_llama_model \
    modelpool=llama_for_causallm \
    taskpool=dummy
```

or using the following configuration file `config/llama_weighted_average.yaml`

```bash
fusion_bench --config-name llama_weighted_average
```

```yaml title="config/llama_weighted_average.yaml"
defaults:
  - example_config
  - override method: weighted_average_for_llama
  - override modelpool: llama_for_causallm
  - _self_

modelpool:
  models:
    # the pre-trained model (base model) is optional
    # if not provided, the first model will be used as the base model
    - name: _pretrained_
      path: ~/data/huggingface_models/meta-llama/Meta-Llama-3-8B
    - name: expert_1
      path: ~/data/huggingface_models/meta-llama/Meta-Llama-3-8B
    - name: expert_2
      path: ~/data/huggingface_models/meta-llama/Meta-Llama-3-8B-Instruct

method:
  normalize: true # if true, the weights will be normalized before merging
  weights: # List of weights for each model
    - 0.4
    - 0.5
  # if true, only the backbone of the model will be merged and the head will be keeped as the pre-trained model (if the pre-trained model is provided, otherwise the head of the first model will be used)
  # if false, the whole model will be merged
  backbone_only: true

  merged_model_save_path: null
  save_tokenizer: true
  push_to_hub: false
```

## References

::: fusion_bench.method.weighted_average.weighted_average.WeightedAverageAlgorithm
    options:
        members: true
::: fusion_bench.method.weighted_average.llama.WeightedAverageForLLama
