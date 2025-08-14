# Flan-T5 Models for Text Generation

## Model Information

Prompt-based fine-tuned Flan-T5 models on GLUE benchmark tasks.
The models are fine-tuned in a text-to-text setting, and the prompt templates are provided below.
The source code for the prompt templates can be found [in the repository](https://github.com/tanganke/fusion_bench/blob/main/fusion_bench/tasks/flan_t5_text_generation/glue_prompt_templates.py).

```python title="fusion_bench/tasks/flan_t5_text_generation/glue_prompt_templates.py"
cola = {
    "description": "template used by GLUE-CoLA",
    "input_text": "Indicate if the following sentence is grammatically correct or not: \"{sentence}\". Answere 'acceptable' or 'unacceptable'.",
    "target_text": {"0": "unacceptable", "1": "acceptable"},
}
mnli = {
    "input_text": "Does the premise: '{premise}' logically imply, contradict, or is neutral to the hypothesis: '{hypothesis}'? Answere with 'entailment', 'contradiction', or 'neutral'.",
    "target_text": {"0": "entailment", "1": "neutral", "2": "contradiction"},
}
mrpc = {
    "input_text": "Are the following sentences '{sentence1}' and '{sentence2}' conveying the same meaning? Answere with 'yes' or 'no'.",
    "target_text": {"0": "no", "1": "yes"},
}
qnli = {
    "input_text": "Given the context: '{sentence}', does the question '{question}' have an answer based on the information provided? Answer with 'yes' or 'no'.",
    "target_text": {"0": "yes", "1": "no"},
}
qqp = {
    "input_text": "Do the questions '{question1}' and '{question2}' have the same intent? Answere with 'yes' or 'no'.",
    "target_text": {"0": "no", "1": "yes"},
}
rte = {
    "description": "Template used by GLUE-RTE",
    "input_text": "Does the text: '{sentence1}' entail that '{sentence2}' is true? Provide 'yes' or 'no'.",
    "target_text": {"0": "yes", "1": "no"},
}
sst2 = {
    "input_text": "Given the sentence '{sentence}', determine the sentiment. Is it positive or negative?",
    "target_text": {"0": "negative", "1": "positive"},
}
stsb = {
    "input_text": "Consider the sentences '{sentence1}' and '{sentence2}'. On a scale from 1 (completely different) to 5 (completely similar), rate the similarity.",
    "target_text": "{:.1f}",
}
```


### Flan-T5-base

#### Full Fine-tuned Models

[full fine-tuned Flan-T5-base models on tasks from GLUE benchmark](https://huggingface.co/collections/tanganke/flan-t5-base-models-fine-tuned-on-glue-benchmark-664f30d7966303d9a0a90bb6)

| Model       | GLUE-COLA     | GLUE-MNLI     | GLUE-MRPC     | GLUE-QNLI     | GLUE-QQP      | GLUE-RTE      | GLUE-SST2     | GLUE-STSB     | Average   |
| ----------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | --------- |
| Pre-trained | 69.127517     | 56.454407     | 76.225490     | 88.449570     | 82.119713     | 80.144404     | 91.169725     | 62.190453     | 75.735160 |
| GLUE-COLA   | **74.976031** | 37.208355     | 72.794118     | 87.552627     | 80.415533     | 76.895307     | 91.399083     | 63.583974     | 73.103128 |
| GLUE-MNLI   | 65.867689     | **83.413143** | 75.735294     | 89.236683     | 82.616869     | 77.978339     | 90.596330     | 66.215025     | 78.957422 |
| GLUE-MRPC   | 63.374880     | 48.293428     | **87.500000** | 85.831960     | 81.100668     | 72.563177     | 88.073394     | 76.062875     | 75.350048 |
| GLUE-QNLI   | 68.744008     | 39.246052     | 75.490196     | **91.488193** | 81.291120     | 78.339350     | 91.628440     | 68.200428     | 74.303474 |
| GLUE-QQP    | 59.060403     | 50.412634     | 73.774510     | 88.339740     | **85.369775** | 81.227437     | 90.825688     | 75.948390     | 75.619822 |
| GLUE-RTE    | 65.388303     | 51.115639     | 69.607843     | 88.705839     | 80.774178     | **85.920578** | 90.252294     | 68.944418     | 75.088636 |
| GLUE-SST2   | 67.785235     | 53.958227     | 76.470588     | 87.772286     | 83.415780     | 80.505415     | **93.577982** | 63.612718     | 75.887279 |
| GLUE-STSB   | 69.319271     | 49.302089     | 76.470588     | 88.962109     | 81.662132     | 77.617329     | 90.137615     | **88.695433** | 77.770821 |

#### LoRA Fine-tuned Models (r=16)

[LoRA fine-tuned (r=16) Flan-T5-base models on tasks from GLUE benchmark](https://huggingface.co/collections/tanganke/flan-t5-base-models-fine-tuned-lora-16-on-glue-benchmark-664eb5848e43ee411fa086f4):

| Model       | GLUE-COLA | GLUE-MNLI | GLUE-MRPC | GLUE-QNLI | GLUE-QQP | GLUE-RTE | GLUE-SST2 | GLUE-STSB | Average |
| ----------- | --------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- | ------- |
| Pre-trained | 69.1      | 56.5      | 76.2      | 88.4      | 82.1     | 80.1     | 91.2      | 62.2      | 75.7    |
| GLUE-COLA   | 69.1      | 39.9      | 75.2      | 89.1      | 81.1     | 81.9     | 90.7      | 54.0      |         |
| GLUE-MNLI   | **69.4**  | **82.7**  | 73.8      | 89.3      | 82.0     | 79.4     | 90.9      | 68.1      |         |
| GLUE-MRPC   | 64.0      | 44.9      | **85.5**  | 82.6      | 81.0     | 69.0     | 88.6      | 73.6      |         |
| GLUE-QNLI   | 68.9      | 52.7      | 76.7      | **90.9**  | 82.8     | 79.8     | 91.5      | 68.9      |         |
| GLUE-QQP    | 65.0      | 54.6      | 75.7      | 89.0      | **84.0** | 81.6     | 90.7      | 75.3      |         |
| GLUE-RTE    | 64.9      | 51.8      | 69.4      | 89.2      | 79.8     | **84.5** | 90.6      | 70.1      |         |
| GLUE-SST2   | 68.3      | 56.6      | 76.0      | 88.5      | 83.4     | 79.8     | **92.9**  | 62.6      |         |
| GLUE-STSB   | 65.7      | 1.7       | 67.4      | 89.3      | 80.1     | 79.8     | 90.8      | **87.4**  |         |

### Flan-T5-Large

#### LoRA Fine-tuned Models (r=16)

[LoRA fine-tuned (r=16) Flan-T5-large models on tasks from GLUE benchmark](https://huggingface.co/collections/tanganke/flan-t5-large-models-fine-tuned-lora-16-on-glue-benchmark-664f2c8835234513c563d087):


| Model      | GLUE-COLA | GLUE-MNLI | GLUE-MRPC | GLUE-QNLI | GLUE-QQP | GLUE-RTE | GLUE-SST2 | GLUE-STSB | Average |
| ---------- | --------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- | ------- |
| Pretrained | 73.7      | 56.6      | 82.4      | 91.1      | 85.5     | 85.6     | 94.3      | 87.5      | 82.1    |
| GLUE-COLA  | **80.2**  | 53.9      | 81.4      | 90.8      | 84.5     | 84.1     | 93.9      | 87.1      |         |
| GLUE-MNLI  | 73.7      | **88.5**  | 77.9      | 92.4      | 85.2     | 87.7     | 94.4      | 86.7      |         |
| GLUE-MRPC  | 75.6      | 52.6      | **89.2**  | 92.6      | 84.4     | 86.3     | 94.3      | 86.3      |         |
| GLUE-QNLI  | 73.5      | 54.5      | 82.8      | **94.4**  | 85.8     | 85.2     | 93.7      | 87.1      |         |
| GLUE-QQP   | 74.0      | 53.8      | 82.8      | 92.5      | **87.2** | 85.6     | 94.5      | 88.3      |         |
| GLUE-RTE   | 75.6      | 57.5      | 69.9      | 92.8      | 83.8     | **91.7** | 94.6      | 86.0      |         |
| GLUE-SST2  | 73.6      | 55.3      | 82.1      | 91.6      | 85.5     | 85.2     | **95.2**  | 86.9      |         |
| GLUE-STSB  | 73.4      | 39.3      | 82.1      | 92.6      | 86.1     | 83.4     | 94.0      | **90.9**  |         |

## Basic Examples

### Inverstigate Model Information

Load pre-trained Flan-T5-base model and print the model information

```bash
fusion_bench \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    method=dummy taskpool=dummy
# {'model_info': {'trainable_params': 247577856, 'all_params': 247577856, 'trainable_percentage': 1.0}}
```

Load pre-trained Flan-T5-large model and print the model information

```bash
fusion_bench \
    modelpool=Seq2SeqLMPool/flan-t5-large_glue_lora16 \
    method=dummy taskpool=dummy
```

### Evaluate Single Model

Evaluate the pre-trained Flan-T5-base model on GLUE tasks

```bash
fusion_bench \
    method=dummy \
    modelpool=Seq2SeqLMPool/flan-t5-base_individual \
        modelpool.models._pretrained_.pretrained_model_name_or_path=google/flan-t5-base \
    taskpool=flan-t5_glue_text_generation \
    report_save_path=outputs/flan-t5-base/pretrained.json
```

or evaluate the fine-tuned Flan-T5-base model on GLUE tasks

```bash
for task in cola mnli mrpc qnli qqp rte sst2 stsb; do
    fusion_bench \
        method=dummy \
        modelpool=Seq2SeqLMPool/flan-t5-base_individual \
            modelpool.models._pretrained_.pretrained_model_name_or_path=tanganke/flan-t5-base_glue-${task} \
        taskpool=flan-t5_glue_text_generation \
        report_save_path=outputs/flan-t5-base/glue-$task.json
done
```

### Simple Average

Merge the Flan-T5 models on GLUE tasks using simple average and evaluate on the Flan-T5 text generation task

```bash
# for full fine-tuned models
fusion_bench \
    method=simple_average \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation

# or using the LoRA fine-tuned models
fusion_bench \
    method=simple_average \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16 \
    taskpool=flan-t5_glue_text_generation
```


### Task Arithmetic

Merge the Flan-T5 models on GLUE tasks using task arithmetic and evaluate on the Flan-T5 text generation task, with scaling factor from 0.0 to 1.0

```bash
# full fine-tuned models with scaling factor set to 0.3
fusion_bench \
    method=task_arithmetic \
        method.scaling_factor=0.3 \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation

# use a for loop to evaluate the performance of task arithmetic with different scaling factors (LoRA fine-tuned models)
for scaling_factor in $(seq 0.0 0.1 1.0)
do
    fusion_bench \
        method=task_arithmetic \
            method.scaling_factor=$scaling_factor \
        modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16 \
        taskpool=flan-t5_glue_text_generation
done
```

| scaling_coef | glue-cola | glue-mnli | glue-mrpc | glue-qnli | glue-qqp | glue-rte | glue-sst2 | glue-stsb | Average  |
| ------------ | --------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- | -------- |
| 0.0          | 69.1      | 56.5      | 76.2      | 88.4      | 82.1     | 80.1     | 91.2      | 62.2      | 75.7     |
| 0.1          | 69.5      | 59.8      | 78.7      | 89.7      | 83.6     | 80.5     | 91.1      | 70.7      | 78.0     |
| 0.2          | 69.3      | 59.0      | 78.7      | 90.1      | 83.8     | 79.1     | 91.5      | 72.9      | **78.1** |
| 0.3          | 68.8      | 55.2      | 78.7      | 89.8      | 83.7     | 79.1     | 91.5      | 72.4      | 77.4     |
| 0.4          | 68.1      | 31.3      | 77.7      | 88.7      | 83.3     | 78.7     | 91.2      | 68.9      | 73.5     |
| 0.5          | 66.0      | 2.2       | 78.7      | 86.3      | 82.6     | 78.0     | 90.4      | 74.2      |          |
| 0.6          | 5.9       | 0.0       | 78.4      | 81.2      | 81.6     | 74.4     | 88.2      | 74.9      |          |
| 0.7          | 0.0       | 0.0       | 77.0      | 74.1      | 79.8     | 66.1     | 70.8      | 74.7      |          |
| 0.8          | 0.0       | 0.0       | 74.0      | 67.5      | 77.0     | 62.1     | 8.1       | 72.5      |          |
| 0.9          | 0.0       | 0.0       | 66.7      | 60.5      | 71.7     | 58.5     | 0.0       | 71.6      |          |
| 1.0          | 0.0       | 0.0       | 46.3      | 50.6      | 56.3     | 52.3     | 0.0       | 69.8      |          |

### Ties-Merging

or using ties-merging

```bash
# for full fine-tuned models with scaling factor set to 0.3
fusion_bench \
    method=ties_merging \
        method.scaling_factor=0.3 \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue \
    taskpool=flan-t5_glue_text_generation

# use a for loop to evaluate the performance of ties-merging with different scaling factors (LoRA fine-tuned models)
for scaling_factor in $(seq 0.0 0.1 1.0)
do
    fusion_bench \
        method=ties_merging \
            method.scaling_factor=$scaling_factor \
        modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16 \
        taskpool=flan-t5_glue_text_generation
done
```

### Experimental Results

Flan-T5-Base models:

=== "Table: Multi-task model merging methods using Flan-T5-Base (full fine-tuned) models."

    | Method                          | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STSB | Average |
    | ------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ------- |
    | Reference Results               |      |      |      |      |      |      |       |      |         |
    | Pre-trained                     | 69.1 | 56.5 | 76.2 | 88.4 | 82.1 | 80.1 | 91.2  | 62.2 | 75.7    |
    | Fine-tuned (STL)                | 75.0 | 83.4 | 87.5 | 91.5 | 85.4 | 85.9 | 93.6  | 88.7 | 86.4    |
    | Model Merging Methods           |      |      |      |      |      |      |       |      |         |
    | Simple Average                  | 69.1 | 62.6 | 79.4 | 89.8 | 83.9 | 81.2 | 91.7  | 73.2 | 78.9    |
    | Task Arithmetic ($\lambda=0.3$) | 70.5 | 57.8 | 78.4 | 90.2 | 83.6 | 80.5 | 92.3  | 77.8 | 78.9    |
    | Ties-Merging ($\lambda=0.3$)    | 70.3 | 65.0 | 78.9 | 90.2 | 83.5 | 81.6 | 91.7  | 78.3 | 79.9    |


=== "Table: Multi-task model merging methods using Flan-T5-Base (LoRA r=16) models."

    | Method                          | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STSB | Average |
    | ------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ------- |
    | Reference Results               |      |      |      |      |      |      |       |      |         |
    | Pre-trained                     | 69.1 | 56.5 | 76.2 | 88.4 | 82.1 | 80.1 | 91.2  | 62.2 | 75.7    |
    | Fine-tuned (STL)                | 69.1 | 82.7 | 85.5 | 90.9 | 84.0 | 84.4 | 92.9  | 87.4 | 84.6    |
    | Model Merging Methods           |      |      |      |      |      |      |       |      |         |
    | Simple Average                  | 69.7 | 59.7 | 78.9 | 90.1 | 83.8 | 90.5 | 91.2  | 72.0 | 78.2    |
    | Task Arithmetic ($\lambda=0.3$) | 68.8 | 55.2 | 78.7 | 89.8 | 83.7 | 79.1 | 91.5  | 72.4 | 77.4    |
    | Ties-Merging ($\lambda=0.3$)    | 68.3 | 56.3 | 79.4 | 89.8 | 83.7 | 79.4 | 91.6  | 71.2 | 77.5    |


Flan-T5-Large models:

=== "Table: Multi-task model merging methods using Flan-T5-Large (LoRA r=16) models"

    | Method                          | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | STSB | Average |
    | ------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ------- |
    | Reference Results               |      |      |      |      |      |      |       |      |         |
    | Pre-trained                     | 73.7 | 56.6 | 82.4 | 91.1 | 85.5 | 85.6 | 94.3  | 87.5 | 82.1    |
    | Fine-tuned (STL)                | 80.2 | 88.5 | 89.2 | 94.4 | 87.2 | 91.7 | 95.2  | 90.9 | 89.6    |
    | Model Merging Methods           |      |      |      |      |      |      |       |      |         |
    | Simple Average                  | 74.6 | 84.3 | 84.1 | 92.8 | 86.3 | 87.4 | 94.8  | 88.0 | 86.5    |
    | Task Arithmetic ($\lambda=0.3$) | 76.9 | 85.4 | 85.3 | 93.9 | 85.8 | 88.1 | 95.2  | 87.8 | 87.3    |
    | Ties-Merging ($\lambda=0.3$)    | 77.1 | 85.1 | 86.3 | 93.9 | 86.0 | 87.7 | 95.1  | 88.0 | 87.4    |


## Implementation Details

- [fusion_bench.modelpool.Seq2SeqLMPool][]
- [fusion_bench.modelpool.SequenceClassificationModelPool][]
- [fusion_bench.modelpool.PeftModelForSeq2SeqLMPool][]
