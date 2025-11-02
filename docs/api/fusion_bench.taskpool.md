# fusion_bench.taskpool

## Base Class

::: fusion_bench.BaseTaskPool

## Vision Task Pool

### NYUv2 Tasks

::: fusion_bench.taskpool.NYUv2TaskPool
    options:
        heading_level: 4

### CLIP Task Pool

::: fusion_bench.taskpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - CLIPVisionModelTaskPool
        - SparseWEMoECLIPVisionModelTaskPool
        - RankoneMoECLIPVisionModelTaskPool

### ResNet for Image Classification

::: fusion_bench.taskpool
    options:
        heading_level: 4
        show_root_heading: false
        members:
        - ResNetForImageClassificationTaskPool

## Natural Language Processing (NLP) Tasks

### GPT-2

::: fusion_bench.taskpool.GPT2TextClassificationTaskPool
    options:
        heading_level: 4

### Flan-T5

::: fusion_bench.compat.taskpool.flan_t5_glue_text_generation.FlanT5GLUETextGenerationTask
    options:
        show_root_full_path: true
        heading_level: 4

### LM-Eval-Harness Integration (LLM)

::: fusion_bench.taskpool.LMEvalHarnessTaskPool
    options:
        heading_level: 4

## Task Agnostic

### Utility Classes

::: fusion_bench.taskpool.DummyTaskPool
    options:
        heading_level: 4
