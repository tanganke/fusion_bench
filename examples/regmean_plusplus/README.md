# RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging

[![arxiv](https://img.shields.io/badge/arXiv-2508.03121-b31b1b.svg)](https://www.arxiv.org/abs/2508.03121)
[![github](https://img.shields.io/badge/GitHub-Code-181717.svg)](https://github.com/nthehai01/RegMean-plusplus)

This directory contains examples demonstrating the **RegMean++** algorithm for merging CLIP vision models on image classification tasks.

## Files Structure

```
regmean_plusplus/
├── README.md           # This file
├── clip_vit.sh         # Script to run RegMean++ on CLIP models
└── results/            # Directory containing pre-run experimental results
    ├── clip-vit-base-patch16/
    ├── clip-vit-base-patch32/
    └── clip-vit-large-patch14/
```

## Usage

### Quick Start

Run the provided script to merge CLIP models using RegMean++:

```bash
bash clip_vit.sh
```

This script will execute RegMean++ on three different CLIP model variants:
- CLIP-ViT-B/32
- CLIP-ViT-B/16
- CLIP-ViT-L/14

## References

1. **RegMean Original Paper**: Xisen Jin, et al. "Dataless Knowledge Fusion by Merging Weights of Language Models." ICLR 2023. [arXiv:2212.09849](https://arxiv.org/abs/2212.09849)

2. **RegMean++ Paper**: The-Hai Nguyen, et al. "RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging." arXiv 2024. [arXiv:2508.03121](https://arxiv.org/abs/2508.03121)

3. **FusionBench**: Comprehensive benchmark for deep model fusion. [GitHub](https://github.com/tanganke/fusion_bench)

## Citation

```bibtex
@article{nguyen2025regmean++,
  title={RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging},
  author={Nguyen, The-Hai and Huu-Tien, Dang and Suzuki, Takeshi and Nguyen, Le-Minh},
  journal={arXiv preprint arXiv:2508.03121},
  year={2025}
}
```

