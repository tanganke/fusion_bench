# GPT-2 Models for Text Classification

Here we provide a series of GPT-2 models fine-tuned for text classification tasks.

## The Seven Tasks from GLUE Benchmark

We provide seven GPT-2 models fine-tuned on the following tasks from GLUE Benchmark:
CoLA, SST-2, MRPC, QQP, MNLI, RTE, and QNLI.
These models are fine-tuned with the learning rate of 5e-5 for 3 epochs.
The models are available on [HuggingFace](https://huggingface.co/collections/tanganke/gpt-2-models-fine-tuned-on-tasks-from-glue-benchmark-664ab37d9e33e622679f541b) as Pytorch models.

Evaluation results of these single-task models on the GLUE Benchmark are as follows:

| Model     | CoLA     | MNLI     | MRPC     | QNLI     | QQP      | RTE      | SST-2    | Avg. |
| --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ---- |
| **CoLA**  | **76.8** | 32.8     | 68.4     | 50.4     | 39.2     | 48.0     | 51.0     | 52.4 |
| **MNLI**  | 59.5     | **82.1** | 33.8     | 46.5     | 24.9     | 57.4     | 40.5     | 49.2 |
| **MRPC**  | 30.8     | 25.9     | **80.4** | 47.1     | 65.9     | 49.1     | 49.1     | 49.8 |
| **QNLI**  | 58.7     | 38.9     | 30.6     | **88.3** | 39.9     | 48.7     | 47.0     | 50.3 |
| **QQP**   | 31.4     | 25.7     | 62.3     | 45.0     | **89.6** | 49.1     | 49.1     | 50.3 |
| **RTE**   | 52.8     | 47.7     | 37.5     | 53.5     | 33.7     | **65.3** | 54.9     | 49.3 |
| **SST-2** | 51.8     | 32.9     | 40.2     | 49.8     | 56.8     | 44.4     | **91.2** | 52.4 |

## Model Pool Configuration

To use these models from our FusionBench library, you can specify the modelpool configuration file as follows:

```yaml title="config/modelpool/gpt-2_glue.yaml"
type: HF_GPT2ForSequenceClassification
models:
  - name: _pretrained_
    path: gpt2
  - name: cola
    path: tanganke/gpt2_cola
  - name: mnli
    path: tanganke/gpt2_mnli
  - name: mrpc
    path: tanganke/gpt2_mrpc
  - name: qnli
    path: tanganke/gpt2_qnli
  - name: qqp
    path: tanganke/gpt2_qqp
  - name: rte
    path: tanganke/gpt2_rte
  - name: sst2
    path: tanganke/gpt2_sst2
```

## Basic Examples

Here are some basic examples of using our CLI tool `fusion_bench` to merge the GPT-2 models.

### Simple Ensemble

construct an ensemble of GPT-2 models using [simple ensemble](../algorithms/simple_ensemble.md) and evaluate on the seven tasks

```bash
fusion_bench method=simple_ensemble \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

### SimpleAverage

merge GPT-2 models using [simple average](../algorithms/simple_averaging.md) and evluate on the seven tasks

```bash
fusion_bench method=simple_average \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

### Fisher merging

merge GPT-2 models using [Fisher Merging](../algorithms/fisher_merging.md) and evluate the merged model

```bash
fusion_bench \
  method=fisher_merging/gpt2_fisher_merging \
    method.batch_size=8 method.num_fisher_examples=512 \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

### RegMean

merge GPT-2 models using [RegMean](../algorithms/regmean.md) and evaluate the merged model

```bash
fusion_bench \
  method=regmean/gpt2_regmean \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
```

### Task Arithmetic

merge using [Task Arithmetic](../algorithms/task_arithmetic.md) on the seven tasks

```bash
# set the scaling factor to 0.3
fusion_bench \
  method=task_arithmetic \
    method.scaling_factor=0.3 \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue

# or run the following script to evaluate the model with different scaling factors,
# and save the results to different files
# or "for scaling_factor in $(seq 0 0.1 1.0)", I use the following for loop for better readability for readers who are not familiar with bash
for scaling_factor in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 
do
fusion_bench report_save_path=outputs/gpt2_glue_task_arithmetic_scaling_factor_${scaling_factor}.json \
  method=task_arithmetic \
    method.scaling_factor=${scaling_factor} \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
done
```

After running the above commands, you will get the following results:

Table: Task Arithmetic with different scaling factors

| scaling_coef | cola     | mnli     | mrpc     | qnli     | qqp      | rte      | sst2     | Avg.         |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ------------ |
| 0.0          | 0.308725 | 0.330107 | 0.313725 | 0.491671 | 0.63166  | 0.527076 | 0.509174 | 0.444591     |
| 0.1          | 0.426654 | 0.501375 | 0.367647 | 0.556654 | 0.739105 | 0.494585 | 0.509174 | 0.513599     |
| 0.2          | 0.658677 | 0.585532 | 0.698529 | 0.602599 | 0.785258 | 0.472924 | 0.669725 | 0.639035     |
| 0.3          | 0.682646 | 0.639837 | 0.718137 | 0.669046 | 0.807915 | 0.462094 | 0.792431 | 0.68173      |
| 0.4          | 0.690316 | 0.673867 | 0.70098  | 0.702178 | 0.817067 | 0.472924 | 0.819954 | 0.696755     |
| 0.5          | 0.68744  | 0.685583 | 0.696078 | 0.704924 | 0.81818  | 0.472924 | 0.836009 | **0.700163** |
| 0.6          | 0.688399 | 0.680998 | 0.678922 | 0.700531 | 0.808978 | 0.472924 | 0.850917 | 0.697381     |
| 0.7          | 0.684564 | 0.665003 | 0.669118 | 0.702361 | 0.789612 | 0.480144 | 0.853211 | 0.692002     |
| 0.8          | 0.677852 | 0.619154 | 0.659314 | 0.673989 | 0.748776 | 0.501805 | 0.819954 | 0.671549     |
| 0.9          | 0.644295 | 0.503515 | 0.654412 | 0.540912 | 0.637942 | 0.487365 | 0.78555  | 0.607713     |
| 1.0          | 0.627996 | 0.411004 | 0.54902  | 0.496614 | 0.478234 | 0.530686 | 0.71445  | 0.544        |

### Ties-Merging

merge using [Ties-Merging](../algorithms/ties_merging.md) on the seven tasks

```bash
fusion_bench \
  method=ties_merging \
    method.scaling_factor=0.3 \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue\

# or run the following script to evaluate the model with different scaling factors,
# and save the results to different files
for scaling_factor in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do fusion_bench report_save_path=outputs/gpt2_glue_ties_merging_scaling_factor_${scaling_factor}.json \
  method=ties_merging \
    method.scaling_factor=${scaling_factor} \
  modelpool=gpt-2_glue \
  taskpool=gpt-2_glue
done
```

| scaling_coef | cola     | mnli     | mrpc     | qnli     | qqp      | rte      | sst2     | Avg.         |
| ------------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ------------ |
| 0.0          | 0.308725 | 0.330107 | 0.313725 | 0.491671 | 0.63166  | 0.527076 | 0.509174 | 0.444591     |
| 0.1          | 0.348035 | 0.45624  | 0.328431 | 0.542559 | 0.70554  | 0.523466 | 0.509174 | 0.487635     |
| 0.2          | 0.489933 | 0.589913 | 0.416667 | 0.596559 | 0.788647 | 0.501805 | 0.510321 | 0.556264     |
| 0.3          | 0.646213 | 0.648497 | 0.632353 | 0.641406 | 0.810611 | 0.516245 | 0.618119 | 0.644778     |
| 0.4          | 0.670182 | 0.691594 | 0.669118 | 0.683141 | 0.821815 | 0.490975 | 0.736239 | 0.680438     |
| 0.5          | 0.681687 | 0.710036 | 0.678922 | 0.696504 | 0.82466  | 0.476534 | 0.77867  | 0.69243      |
| 0.6          | 0.683605 | 0.713805 | 0.683824 | 0.695589 | 0.823967 | 0.476534 | 0.817661 | **0.699284** |
| 0.7          | 0.685523 | 0.700968 | 0.64951  | 0.689365 | 0.816893 | 0.487365 | 0.829128 | 0.694107     |
| 0.8          | 0.686481 | 0.68538  | 0.64951  | 0.693209 | 0.801608 | 0.483755 | 0.837156 | 0.691014     |
| 0.9          | 0.684564 | 0.650229 | 0.671569 | 0.69687  | 0.775587 | 0.516245 | 0.833716 | 0.689826     |
| 1.0          | 0.667306 | 0.576566 | 0.661765 | 0.645616 | 0.72372  | 0.490975 | 0.822248 | 0.655456     |


### Experimental Results

Table: Multi-task model merging methods using GPT-2 models

| Method                          | CoLA | MNLI | MRPC | QNLI | QQP  | RTE  | SST-2 | Avg. |
| ------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- |
| Fine-tuned (STL)                | 76.8 | 82.1 | 80.4 | 88.3 | 89.6 | 65.3 | 91.2  | 82.0 |
| Model Merging                   |
| Simple Average                  | 55.0 | 55.1 | 51.0 | 57.6 | 76.7 | 44.8 | 52.5  | 56.1 |
| Fisher Merging                  | 54.8 | 58.0 | 39.5 | 63.3 | 81.5 | 49.1 | 64.7  | 58.7 |
| RegMean                         | 61.7 | 70.4 | 65.4 | 69.7 | 78.8 | 56.0 | 79.7  | 68.8 |
| Task Arithmetic ($\lambda=0.5$) | 68.7 | 68.6 | 69.6 | 70.5 | 81.8 | 47.3 | 83.6  | 70.0 |
| Ties-Merging ($\lambda=0.6$)    | 68.4 | 71.4 | 68.4 | 69.6 | 82.4 | 47.7 | 81.8  | 70.0 |
