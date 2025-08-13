# fusion_bench.tasks

## Image Classification Tasks

::: fusion_bench.tasks.clip_classification
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3
        members:
        - get_classnames_and_templates
        - CLIPTemplateFactory

## Flan-T5 Text Generation Tasks

::: fusion_bench.tasks.flan_t5_text_generation.glue_preprocessors
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

::: fusion_bench.tasks.flan_t5_text_generation.glue_load_dataset
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3
        members:
        - load_glue_dataset

::: fusion_bench.tasks.flan_t5_text_generation.glue_evaluation
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

::: fusion_bench.tasks.flan_t5_text_generation.glue_prompt_templates
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3
        members:
        - glue_prompt_templates
        - cola
        - mnli
        - mrpc
        - qnli
        - qqp
        - rte
        - stsb
        - sst2
