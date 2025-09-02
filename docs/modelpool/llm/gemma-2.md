# Gemma-2

## Gemma-2-2B Models

This configuration includes the base model and specialized fine-tuned variants from MergeBench:

```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-2b.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-2b.yaml"
```

This configuration focuses on instruction-tuned variants:

```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-2b-it.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-2b-it.yaml"
```

### Model Fusion Experiments

#### Simple Average

```shell
fusion_bench path.log_dir=outputs/gemma-2-2b/simple_average \
    method=linear/simple_average_for_causallm \
    modelpool=CausalLMPool/mergebench/gemma-2-2b
```

## Gemma-2-9B Models

This configuration includes the base model and specialized fine-tuned variants from MergeBench:

```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-9b.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-9b.yaml"
```

This configuration focuses on instruction-tuned variants:

```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-9b-it.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-9b-it.yaml"
```
