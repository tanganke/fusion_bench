# Reading Lists

!!! info
    working in progress

I've been compiling a comprehensive list of papers and resources that have been instrumental in my research journey. 
This collection is designed to serve as a valuable starting point for those interested in delving into the field of deep model fusion.
*If you have any suggestions for papers to add, please feel free to raise an issue or submit a pull request.*

!!! note

    Meaning of the symbols in the list:

    - :star:  Highly recommended
    - :llama: LLaMA model-related
    - :simple-github: Code available on GitHub
    - :hugging: models or datasets available on Hugging Face

## Survey Papers

- :star: 
    W. Li, Y. Peng, M. Zhang, L. Ding, H. Hu, and L. Shen, “Deep Model Fusion: A Survey.” [arXiv, Sep. 27, 2023. doi: 10.48550/arXiv.2309.15698.](http://arxiv.org/abs/2309.15698)
- [:simple-github:](https://github.com/ruthless-man/Awesome-Learn-from-Model) 
    H. Zheng et al., “Learn From Model Beyond Fine-Tuning: A Survey.” [arXiv, Oct. 12, 2023.](http://arxiv.org/abs/2310.08184)

## Model Ensemble

## Model Merging

### Mode Connectivity

Mode connectivity is such an important concept in model merging that it deserves [its own page](mode_connectivity.md).

### Weight Interpolation

- G. Ilharco et al., “Editing Models with Task Arithmetic,” Mar. 31, 2023, arXiv: arXiv:2212.04089. doi: 10.48550/arXiv.2212.04089.
- Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard, “Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models,” May 30, 2023, arXiv: arXiv:2305.12827. doi: 10.48550/arXiv.2305.12827.
- P. Yadav, D. Tam, L. Choshen, C. Raffel, and M. Bansal, “Resolving Interference When Merging Models,” Jun. 02, 2023, arXiv: arXiv:2306.01708. Accessed: Jun. 12, 2023. [Online]. Available: http://arxiv.org/abs/2306.01708
- [:simple-github:](https://github.com/EnnengYang/AdaMerging) 
    E. Yang et al., “AdaMerging: Adaptive Model Merging for Multi-Task Learning,” ICLR 2024, arXiv: arXiv:2310.02575. doi: 10.48550/arXiv.2310.02575.
- :llama: [:simple-github:](https://github.com/yule-BUAA/MergeLM)
    L. Yu, B. Yu, H. Yu, F. Huang, and Y. Li, “Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch,” Nov. 06, 2023, arXiv: arXiv:2311.03099. Accessed: Nov. 07, 2023. [Online]. Available: http://arxiv.org/abs/2311.03099


### Alignment-based Methods

- S. K. Ainsworth, J. Hayase, and S. Srinivasa, “Git Re-Basin: Merging Models modulo Permutation Symmetries,” ICLR 2023. Available: http://arxiv.org/abs/2209.04836
- George Stoica, Daniel Bolya, Jakob Bjorner, Taylor Hearn, and Judy Hoffman, “ZipIt! Merging Models from Different Tasks without Training,” May 04, 2023, arXiv: arXiv:2305.03053. Accessed: May 06, 2023. [Online]. Available: http://arxiv.org/abs/2305.03053


### Subspace-based Methods

- :llama: [:simple-github:](https://github.com/xinykou/safety_realignment) 
    X. Yi, S. Zheng, L. Wang, X. Wang, and L. He, “A safety realignment framework via subspace-oriented model fusion for large language models.” [arXiv, May 14, 2024. doi: 10.48550/arXiv.2405.09055.](http://arxiv.org/abs/2405.09055)

## Model Mixing

- :llama: [:simple-github:](https://github.com/THUNLP-MT/ModelCompose) :hugging:
    C. Chen et al., “Model Composition for Multimodal Large Language Models.” [arXiv, Feb. 20, 2024. doi: 10.48550/arXiv.2402.12750.](http://arxiv.org/abs/2402.12750)
- A. Tang, L. Shen, Y. Luo, N. Yin, L. Zhang, and D. Tao, “Merging Multi-Task Models via Weight-Ensembling Mixture of Experts,” Feb. 01, 2024, arXiv: arXiv:2402.00433. doi: 10.48550/arXiv.2402.00433.
- :llama: [:simple-github:](https://github.com/LZY-the-boys/Twin-Merging) 
    Zhenyi Lu et al., "Twin-Merging: Dynamic Integration of Modular Expertise in Model Merging" [10.48550/arXiv.2406.15479](http://arxiv.org/abs/2406.15479)


## Libraries and Tools

### Fine-tuning, Preparing models for fusion

- [:simple-github:](https://github.com/tanganke/pytorch_classification)
    PyTorch Classification: A PyTorch library for training/fine-tuning models (CNN, ViT, CLIP) on image classification tasks
- :star: [:simple-github:](https://github.com/hiyouga/LLaMA-Factory)
    LLaMA Factory: A PyTorch library for fine-tuning LLMs

### Model Fusion

- :star: [:simple-github:](https://github.com/tanganke/fusion_bench) [:hugging:](https://huggingface.co/tanganke)
    FusionBench: A Comprehensive Benchmark of Deep Model Fusion.
- :star: :llama: [:simple-github:](https://github.com/arcee-ai/mergekit) 
    MergeKit: A PyTorch library for merging large language models.
