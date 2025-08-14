# Model Training/Fine-Tuning

## CLIP vision model fine-tuning

- [ImageClassificationFineTuningForCLIP][fusion_bench.method.ImageClassificationFineTuningForCLIP]: Fine-tuning clip vision encoder on image classification tasks.
- [ContinualImageClassificationFineTuningForCLIP][fusion_bench.method.ContinualImageClassificationFineTuningForCLIP]: Continual fine-tuning of clip vision encoder on image classification tasks.

::: fusion_bench.method
    options:
        show_root_heading: false
        members:
            - ImageClassificationFineTuningForCLIP
            - ContinualImageClassificationFineTuningForCLIP


## LLM Fine-tuning

::: fusion_bench.method
    options:
        show_root_heading: false
        members:
        - FullFinetuneSFT
        - PeftFinetuneSFT

## Reward Modeling

::: fusion_bench.method
    options:
        show_root_heading: false
        members:
        - BradleyTerryRewardModeling
