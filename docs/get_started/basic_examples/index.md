# Basic Examples

Start here to learn the fundamentals of FusionBench through hands-on examples.

## Getting Started with FusionBench CLI

The quickest way to get started is to run your first fusion experiment:

```bash
fusion_bench \
    method=simple_average \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

<div class="grid cards" markdown style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem;">

- **Structured Configs**

    ---

    Learn how to build structured configuration files and group configurations effectively in FusionBench.

    [:octicons-arrow-right-24: Read More](structured_configs.md)

- **CLIP Simple Average**

    ---

    Merge clip vision models using simple average.

    [:octicons-arrow-right-24: Read More](clip_simple_average.md)

- **CLIP Task Arithmetic**

    ---

    Merge CLIP vision models using task arithmetic, allowing you to adjust the scaling factor as a hyperparameter.

    [:octicons-arrow-right-24: Read More](clip_task_arithmetic.md)

- **Evaluate Single CLIP Model**
    
    ---

    Evaluate the performance of a single CLIP model on image classification tasks.

    [:octicons-arrow-right-24: Read More](evaluate_single_clip_model.md)


- **Merge Large Language Models**

    ---

    Merge large language models using SLERP.

    [:octicons-arrow-right-24: Read More](merge_llm.md)


</div>

## FusionBench as a Package

<div class="grid cards" markdown style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem;">

- **Import and Use Merging Algorithms**

    ---

    Learn how to import and use different merging algorithms in FusionBench.

    [:octicons-arrow-right-24: Read More](import_and_use_merging_algorithms.md)

- **Parallel Ensemble**

    ---

    Learn how to create an ensemble from multiple CLIP vision models and inference in parallel using FusionBench.

    [:octicons-arrow-right-24: Read More](parallel_clip_ensemble.md)

</div>

## Prerequisites

Before running these examples, make sure you:

1. Have FusionBench installed: `pip install fusion-bench`
2. Have PyTorch and Transformers installed.
3. Familiarize yourself with [Hydra basics](https://hydra.cc/docs/intro/)
