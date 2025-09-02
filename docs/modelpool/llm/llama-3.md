# LLaMA-3

## Llama-3.1-8B

This configuration includes the pretrained base model along with domain-specific fine-tuned models from MergeBench:

```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B.yaml"
```

This configuration focuses on instruction-tuned variants:

```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B-Instruct.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B-Instruct.yaml"
```

### Model Fusion Experiments

#### Simple Average

```shell
fusion_bench path.log_dir=outputs/llama-3.1-8b/simple_average \
    method=linear/simple_average_for_causallm \
    modelpool=CausalLMPool/mergebench/Llama-3.1-8B
```

## Llama-3.2-3B

This configuration includes the pretrained base model along with domain-specific fine-tuned models from MergeBench:

```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B.yaml"
```

This configuration focuses on instruction-tuned variants:

```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B-Instruct.yaml"
--8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B-Instruct.yaml"
```

### Model Fusion Experiments

#### Simple Average

```shell
fusion_bench path.log_dir=outputs/llama-3.2-3b/simple_average \
    method=linear/simple_average_for_causallm \
    modelpool=CausalLMPool/mergebench/Llama-3.2-3B
```
