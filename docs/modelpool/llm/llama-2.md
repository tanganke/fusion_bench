# LLaMA-2

## LLaMA-2-7B Models

```yaml title="config/modelpool/CausalLMPool/llama-7b_3-models_v1.yaml"
--8<-- "config/modelpool/CausalLMPool/llama-7b_3-models_v1.yaml"
```

### Model Fusion Experiments

#### Simple Average

```shell
fusion_bench path.log_dir=outputs/llama-2/3-models_v1/simple_average \
    method=linear/simple_average_for_causallm modelpool=CausalLMPool/llama-7b_3-models_v1
```
