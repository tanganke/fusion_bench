# fusion_bench.modelpool

## Base Class

::: fusion_bench.BaseModelPool

## Vision Model Pool

### NYUv2 Tasks (ResNet)

::: fusion_bench.modelpool.NYUv2ModelPool
    options:
        heading_level: 4

### CLIP Vision Encoder

::: fusion_bench.modelpool.CLIPVisionModelPool
    options:
        heading_level: 4

### OpenCLIP Vision Encoder

::: fusion_bench.modelpool.OpenCLIPVisionModelPool
    options:
        heading_level: 4

## NLP Model Pool

### GPT-2

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - HuggingFaceGPT2ClassificationPool
        - GPT2ForSequenceClassificationPool
  
## Seq2Seq Language Models (Flan-T5)

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - Seq2SeqLMPool
        - SequenceClassificationModelPool
        - PeftModelForSeq2SeqLMPool

## Causal Language Models (Llama, Mistral, Qwen...)

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

## Others

### Transformers AutoModel

::: fusion_bench.modelpool
    options:
        show_root_heading: false
        heading_level: 4
        members:
        - AutoModelPool
