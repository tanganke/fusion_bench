# fusion_bench.modelpool

## Base Class

### BaseModelPool

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - BaseModelPool

## Vision Model Pools

### CLIP Vision Encoder

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - CLIPVisionModelPool

### OpenCLIP Vision Encoder

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - OpenCLIPVisionModelPool

### NYUv2 Model Pool

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - NYUv2ModelPool

### ResNet for Image Classification

::: fusion_bench.modelpool.resnet_for_image_classification
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - load_torchvision_resnet
        - load_transformers_resnet
        - ResNetForImageClassificationPool

### ConvNeXt for Image Classification

::: fusion_bench.modelpool.convnext_for_image_classification
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - load_transformers_convnext
        - ConvNextForImageClassificationPool

### DINOv2 for Image Classification

::: fusion_bench.modelpool.dinov2_for_image_classification
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - load_transformers_dinov2
        - Dinov2ForImageClassificationPool

## NLP Model Pools

### GPT-2 for Sequence Classification

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - HuggingFaceGPT2ClassificationPool
        - GPT2ForSequenceClassificationPool

### Seq2Seq Language Models (Flan-T5)

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - Seq2SeqLMPool
        - PeftModelForSeq2SeqLMPool

### Sequence Classification Language Models

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - SequenceClassificationModelPool

### Causal Language Models (Llama, Mistral, Qwen...)

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - CausalLMPool
        - CausalLMBackbonePool

::: fusion_bench.modelpool.causal_lm.load_peft_causal_lm
    options:
        heading_level: 4

## Generic Model Pools

### Transformers AutoModel

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - AutoModelPool
