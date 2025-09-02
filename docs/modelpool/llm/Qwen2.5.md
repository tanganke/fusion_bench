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

In FusionBench, we provide several pre-configured model pools for Qwen2.5-1.5B models that are commonly used for model fusion experiments. 
These configurations include base models and their fine-tuned variants specialized for different domains.

This configuration includes the base model along with instruction, math, and code variants:

```yaml title="config/modelpool/CausalLMPool/Qwen2.5-1.5B_three_models.yaml"
--8<-- "config/modelpool/CausalLMPool/Qwen2.5-1.5B_three_models.yaml"
```

This configuration focuses specifically on mathematical and coding capabilities:

```yaml title="config/modelpool/CausalLMPool/Qwen2.5-1.5B_math_and_code.yaml"
--8<-- "config/modelpool/CausalLMPool/Qwen2.5-1.5B_math_and_code.yaml"
```

### Model Fusion Experiments

#### Simple Average

Merge all three specialized models using simple parameter averaging:

```shell
fusion_bench path.log_dir=outputs/Qwen2.5-1.5B/three_models/simple_average \
    method=linear/simple_average_for_causallm \
    modelpool=CausalLMPool/Qwen2.5-1.5B_three_models
```

> Example for evaluating the merged model using lm-eval-harness on gsm8k and gsm8k_cot tasks:
> 
> ```shell
> scripts/lm_eval/evaluate_task.sh \
>     outputs/Qwen2.5-1.5B/three_models/simple_average/checkpoint \
>     --tasks 'gsm8k,gsm8k_cot' --output_path outputs/lm_eval
> ```

Merge math and code models using simple parameter averaging:

```shell
fusion_bench path.log_dir=outputs/Qwen2.5-1.5B/math_and_code/simple_average \
    method=linear/simple_average_for_causallm \
    modelpool=CausalLMPool/Qwen2.5-1.5B_math_and_code
```

#### Task Arithmetic

Merge all three specialized models using task arithmetic:

```shell
scaling_factor=0.8
fusion_bench path.log_dir=outputs/Qwen2.5-1.5B/three_models/task_arithmetic/${scaling_factor} \
    method=linear/task_arithmetic_for_causallm \
    method.scaling_factor=${scaling_factor} \
    modelpool=CausalLMPool/Qwen2.5-1.5B_three_models
```

#### Ties-Merging

Merge all three specialized models using TIES merging:

```shell
scaling_factor=0.8
fusion_bench path.log_dir=outputs/Qwen2.5-1.5B/three_models/ties_merging/${scaling_factor} \
    method=linear/ties_merging_for_causallm \
    method.scaling_factor=${scaling_factor} \
    modelpool=CausalLMPool/Qwen2.5-1.5B_three_models
```

## Citation

If you use Qwen2.5 models in your research, please cite:

```bibtex
@misc{qwen2025qwen25technicalreport,
      title={Qwen2.5 Technical Report}, 
      author={Qwen and : and An Yang and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chengyuan Li and Dayiheng Liu and Fei Huang and Haoran Wei and Huan Lin and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jingren Zhou and Junyang Lin and Kai Dang and Keming Lu and Keqin Bao and Kexin Yang and Le Yu and Mei Li and Mingfeng Xue and Pei Zhang and Qin Zhu and Rui Men and Runji Lin and Tianhao Li and Tianyi Tang and Tingyu Xia and Xingzhang Ren and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yu Wan and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zihan Qiu},
      year={2025},
      eprint={2412.15115},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.15115}, 
}
```
