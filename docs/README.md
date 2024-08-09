---
title: FusionBench
description: A Comprehensive Benchmark of Deep Model Fusion
---

# FusionBench: A Comprehensive Benchmark of Deep Model Fusion

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](http://arxiv.org/abs/2406.03280)
[![Downloads](https://static.pepy.tech/badge/fusion-bench/month)](https://pepy.tech/project/fusion-bench)
[![Downloads](https://static.pepy.tech/badge/fusion-bench/week)](https://pepy.tech/project/fusion-bench)

!!! warning "Testing Phase"
    
    The documented experiments should be reproducible. 
    However, this project is still in testing phase as the API may be subject to change. 
    Please report any issues you encounter.

!!! note
    
    - Any questions or comments can be directed to the [GitHub Issues](https://github.com/tanganke/fusion_bench/issues) page for this project.
    - Any contributions or pull requests are welcome. If you find any mistakes or have suggestions for improvements, please feel free to raise an issue or submit a pull request.

!!! tip "Introduction to Deep Model Fusion (The Learn From Model Paradigm)"

    Deep model fusion is a technique that merges, ensemble, or fuse multiple deep neural networks to obtain a unified model.
    It can be used to improve the performance and robustness of model or to combine the strengths of different models, such as fuse multiple task-specific models to create a multi-task model.
    For a more detailed introduction to deep model fusion, you can refer to [W. Li, 2023, 'Deep Model Fusion: A Survey'](https://arxiv.org/abs/2309.15698). 
    In this benchmark, we evaluate the performance of different fusion methods on a variety of datasets and tasks. ...

    [:octicons-arrow-right-24: Read More](introduction_to_model_fusion.md)

## Getting Started

### Installation

install from [PyPI](https://pypi.org/project/fusion-bench/):

```bash
pip install fusion-bench
```

or install the latest version in development from github repository

```bash
git clone https://github.com/tanganke/fusion_bench.git
cd fusion_bench

pip install -e . # install the package in editable mode
```

### Command Line Interface

`fusion_bench` is the command line interface for running the benchmark. 
It takes a configuration file as input, which specifies the models, fusion method to be used, and the datasets to be evaluated. 
To run the benchmark, you can use the following command:

```
fusion_bench [--config-path CONFIG_PATH] [--config-name CONFIG_NAME] \
    OPTION_1=VALUE_1 OPTION_2=VALUE_2 ...
```

This program will load the configuration file specified by `--config-path` and `--config-name`, and run the fusion algorithm on the model pool.
The pseudocode is as follows:

```python
# instantiate an algorithm, a modelpool object that manages the models, 
# and a taskpool object that manages the tasks (dataset + metrics)
algorithm = load_algorithm(config.algorithm)
modelpool = load_modelpool(config.modelpool)
taskpool = load_taskpool(config.taskpool)

# run the fusion algorithm on the model pool
merged_model = algorithm.run(modelpool)
# evaluate the merged model on the tasks
report = taskpool.evaluate(merged_model)
```

For detailed information on the options available, you can refer to this [page](cli/fusion_bench.md).

## General Structure of FusionBench

<figure markdown="span">
![alt text](images/framework_of_model_fusion.png){ width="800px" }
<figcaption>Framework of FusionBench</figcaption>
</figure>

FusionBench is a pioneering project that provides a comprehensive benchmark for deep model fusion, facilitating the evaluation and comparison of various model fusion techniques. The project is meticulously designed to support rigorous analysis and experimentation in the field of model fusion, offering a versatile and modular codebase tailored for advanced research and development.

The general structure of the FusionBench project can be visualized through its modular framework, which is divided into several key components:

1. **Fusion Algorithm**: The core component where Model Fusion takes place. It integrates models from the Model Pool and adjusts them according to the specified fusion algorithms. The output is then evaluated for performance and effectiveness.
2. **Model Pool**: A repository of various pre-trained models that can be accessed and utilized for fusion. This pool serves as the foundation for creating new, fused models by leveraging the strengths of each individual model.
3. **Task Pool**: A collection of tasks that the fused models are evaluated on. These tasks help in assessing the practical applicability and robustness of the fused models.
4. **Models & Warpers, Datasets, and Metrics**: These underlying modules include:
      - Models & Warpers: Tools and scripts for model loading, wrapping, and pre-processing.
      - Datasets: The datasets used for training, validation, and testing the fused models.
      - Metrics: The performance metrics used to evaluate the models, providing a comprehensive understanding of their capabilities.
5. **YAML Configurations**: Central to the project's modularity, YAML files are used to configure models, datasets, and metrics, allowing seamless customization and scalability.

By organizing these components into a structured and modular codebase, FusionBench ensures flexibility, ease of use, and scalability for researchers and developers. The project not only serves as a benchmark but also as a robust platform for innovation in the realm of deep model fusion.

<div class="grid cards" markdown>

- **Fusion Algorithm Module**
    
    ---

    Implement the fusion algorithms. Receive the model pool and return the fused model.

    [:octicons-arrow-right-24: Read More](algorithms/README.md)

- **Model Pool Module**

    ---

     Magage the models. Responsible for loading, preprocessing, and saving the models.
        
    [:octicons-arrow-right-24: Read More](modelpool/README.md)

- **Task Pool Module**

    ---

    Manage the tasks. Responsible for loading evaluation datasets and metrics, and evaluating the fused model.

    [:octicons-arrow-right-24: Read More](taskpool/README.md)

</div>

## Citation

If you find this benchmark useful, please consider citing our work:

```bibtex
@misc{tangFusionBenchComprehensiveBenchmark2024,
  title = {{{FusionBench}}: {{A Comprehensive Benchmark}} of {{Deep Model Fusion}}},
  shorttitle = {{{FusionBench}}},
  author = {Tang, Anke and Shen, Li and Luo, Yong and Hu, Han and Du, Bo and Tao, Dacheng},
  year = {2024},
  month = jun,
  number = {arXiv:2406.03280},
  eprint = {2406.03280},
  publisher = {arXiv},
  url = {http://arxiv.org/abs/2406.03280},
  archiveprefix = {arxiv},
  langid = {english},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Computation and Language,Computer Science - Machine Learning}
}
```
