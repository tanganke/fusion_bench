# Flan-T5 Models for Text Generation

[LoRA fine-tuned (r=16) Flan-T5-base models on tasks from GLUE benchmark](https://huggingface.co/collections/tanganke/flan-t5-base-models-fine-tuned-lora-16-on-glue-benchmark-664eb5848e43ee411fa086f4):

| Model      | GLUE-COLA | GLUE-MNLI | GLUE-MRPC | GLUE-QNLI | GLUE-QQP | GLUE-RTE | GLUE-SST2 | GLUE-STSB |
| ---------- | --------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- |
| Pretrained | 69.1      | 56.5      | 76.2      | 88.4      | 82.1     | 80.1     | 91.2      | 62.2      |
| GLUE-COLA  | 69.1      | 39.9      | 75.2      | 89.1      | 81.1     | 81.9     | 90.7      | 54.0      |
| GLUE-MNLI  | **69.4**  | **82.7**  | 73.8      | 89.3      | 82.0     | 79.4     | 90.9      | 68.1      |
| GLUE-MRPC  | 64.0      | 44.9      | **85.5**  | 82.6      | 81.0     | 69.0     | 88.6      | 73.6      |
| GLUE-QNLI  | 68.9      | 52.7      | 76.7      | **90.9**  | 82.8     | 79.8     | 91.5      | 68.9      |
| GLUE-QQP   | 65.0      | 54.6      | 75.7      | 89.0      | **84.0** | 81.6     | 90.7      | 75.3      |
| GLUE-RTE   | 64.9      | 51.8      | 69.4      | 89.2      | 79.8     | **84.5** | 90.6      | 70.1      |
| GLUE-SST2  | 68.3      | 56.6      | 76.0      | 88.5      | 83.4     | 79.8     | **92.9**  | 62.6      |
| GLUE-STSB  | 65.7      | 1.7       | 67.4      | 89.3      | 80.1     | 79.8     | 90.8      | **87.4**  |

[LoRA fine-tuned (r=16) Flan-T5-large models on tasks from GLUE benchmark](https://huggingface.co/collections/tanganke/flan-t5-large-models-fine-tuned-lora-16-on-glue-benchmark-664f2c8835234513c563d087):


| Model      | GLUE-COLA | GLUE-MNLI | GLUE-MRPC | GLUE-QNLI | GLUE-QQP | GLUE-RTE | GLUE-SST2 | GLUE-STSB |
| ---------- | --------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- |
| Pretrained | 73.7      | 56.6      | 82.4      | 91.1      | 85.5     | 85.6     | 94.3      | 87.5      |
| GLUE-COLA  | **80.2**  | 53.9      | 81.4      | 90.8      | 84.5     | 84.1     | 93.9      | 87.1      |
| GLUE-MNLI  | 73.7      | **88.5**  | 77.9      | 92.4      | 85.2     | 87.7     | 94.4      | 86.7      |
| GLUE-MRPC  | 75.6      | 52.6      | **89.2**  | 92.6      | 84.4     | 86.3     | 94.3      | 86.3      |
| GLUE-QNLI  | 73.5      | 54.5      | 82.8      | **94.4**  | 85.8     | 85.2     | 93.7      | 87.1      |
| GLUE-QQP   | 74.0      | 53.8      | 82.8      | 92.5      | **87.2** | 85.6     | 94.5      | 88.3      |
| GLUE-RTE   | 75.6      | 57.5      | 69.9      | 92.8      | 83.8     | **91.7** | 94.6      | 86.0      |
| GLUE-SST2  | 73.6      | 55.3      | 82.1      | 91.6      | 85.5     | 85.2     | **95.2**  | 86.9      |
| GLUE-STSB  | 73.4      | 39.3      | 82.1      | 92.6      | 86.1     | 83.4     | 94.0      | **90.9**  |

## Examples

Merge the Flan-T5 models on GLUE tasks using simple average and evaluate on the Flan-T5 text generation task

```bash
fusion_bench \
    method=simple_average \
    modelpool=flan-t5-base_glue_lora16 \
    taskpool=flan-t5_glue_text_generation
```

| glue-cola | glue-mnli | glue-mrpc | glue-qnli | glue-qqp | glue-rte | glue-sst2 | glue-stsb |
| --------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- |
| 69.7      | 59.7      | 78.9      | 90.1      | 83.8     | 90.5     | 91.2      | 72.0      |


Merge the Flan-T5 models on GLUE tasks using task arithmetic and evaluate on the Flan-T5 text generation task, with scaling factor from 0.0 to 1.0

```bash
for scaling_factor in $(seq 0.0 0.1 1.0)
do
    fusion_bench \
        method=task_arithmetic \
            method.scaling_factor=$scaling_factor \
        modelpool=flan-t5-base_glue_lora16 \
        taskpool=flan-t5_glue_text_generation
done
```

| scaling_coef | glue-cola | glue-mnli | glue-mrpc | glue-qnli | glue-qqp | glue-rte | glue-sst2 | glue-stsb |
| ------------ | --------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- |
| 0.0          | 69.1      | 56.5      | 76.2      | 88.4      | 82.1     | 80.1     | 91.2      | 62.2      |
| 0.1          | 69.5      | 59.8      | 78.7      | 89.7      | 83.6     | 80.5     | 91.1      | 70.7      |
| 0.2          | 69.3      | 59.0      | 78.7      | 90.1      | 83.8     | 79.1     | 91.5      | 72.9      |
| 0.3          | 68.8      | 55.2      | 78.7      | 89.8      | 83.7     | 79.1     | 91.5      | 72.4      |
| 0.4          | 68.1      | 31.3      | 77.7      | 88.7      | 83.3     | 78.7     | 91.2      | 68.9      |
| 0.5          | 66.0      | 2.2       | 78.7      | 86.3      | 82.6     | 78.0     | 90.4      | 74.2      |
| 0.6          | 5.9       | 0.0       | 78.4      | 81.2      | 81.6     | 74.4     | 88.2      | 74.9      |
| 0.7          | 0.0       | 0.0       | 77.0      | 74.1      | 79.8     | 66.1     | 70.8      | 74.7      |
| 0.8          | 0.0       | 0.0       | 74.0      | 67.5      | 77.0     | 62.1     | 8.1       | 72.5      |
| 0.9          | 0.0       | 0.0       | 66.7      | 60.5      | 71.7     | 58.5     | 0.0       | 71.6      |
| 1.0          | 0.0       | 0.0       | 46.3      | 50.6      | 56.3     | 52.3     | 0.0       | 69.8      |

or using ties-merging

```bash
for scaling_factor in $(seq 0.0 0.1 1.0)
do
    fusion_bench \
        method=ties_merging \
            method.scaling_factor=$scaling_factor \
        modelpool=flan-t5-base_glue_lora16 \
        taskpool=flan-t5_glue_text_generation
done
```

### References

::: fusion_bench.modelpool.AutoModelForSeq2SeqLM.AutoModelForSeq2SeqLMPool

::: fusion_bench.modelpool.PeftModelForSeq2SeqLM.PeftModelForSeq2SeqLMPool
