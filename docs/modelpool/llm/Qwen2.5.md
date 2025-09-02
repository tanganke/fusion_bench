# Qwen2.5

Qwen2.5 is a series of large language models developed by Alibaba Cloud's Qwen team. These models are designed to excel in various natural language processing tasks including text generation, code completion, mathematical reasoning, and more. The Qwen2.5 series offers models of different sizes to accommodate various computational requirements and use cases.

The following table shows the architecture details and licensing information for all Qwen2.5 open-weight models:

| Models | Layers | Heads (Q / KV) | Tie Embedding | Context / Generation Length | License       |
| ------ | ------ | -------------- | ------------- | --------------------------- | ------------- |
| 0.5B   | 24     | 14 / 2         | Yes           | 32K / 8K                    | Apache 2.0    |
| 1.5B   | 28     | 12 / 2         | Yes           | 32K / 8K                    | Apache 2.0    |
| 3B     | 36     | 16 / 2         | Yes           | 32K / 8K                    | Qwen Research |
| 7B     | 28     | 28 / 4         | No            | 128K / 8K                   | Apache 2.0    |
| 14B    | 48     | 40 / 8         | No            | 128K / 8K                   | Apache 2.0    |
| 32B    | 64     | 40 / 8         | No            | 128K / 8K                   | Apache 2.0    |
| 72B    | 80     | 64 / 8         | No            | 128K / 8K                   | Qwen          |

## Qwen2.5-1.5B Models

In FusionBench, we provide several pre-configured model pools for Qwen2.5-1.5B models that are commonly used for model fusion experiments. These configurations include base models and their fine-tuned variants specialized for different domains.

```yaml title="config/modelpool/CausalLMPool/Qwen2.5-1.5B_three_models.yaml"
--8<-- "config/modelpool/CausalLMPool/Qwen2.5-1.5B_three_models.yaml"
```

```yaml title="config/modelpool/CausalLMPool/Qwen2.5-1.5B_math_and_code.yaml"
--8<-- "config/modelpool/CausalLMPool/Qwen2.5-1.5B_math_and_code.yaml"
```

### Model Fusion Strategies

#### Simple Average

```shell
fusion_bench path.log_dir=outputs/Qwen2.5-1.5B/three_models/simple_average \
    method=linear/simple_average_for_causallm modelpool=CausalLMPool/Qwen2.5-1.5B_three_models
```

```shell
fusion_bench path.log_dir=outputs/Qwen2.5-1.5B/math_and_code/simple_average \
    method=linear/simple_average_for_causallm modelpool=CausalLMPool/Qwen2.5-1.5B_math_and_code
```
