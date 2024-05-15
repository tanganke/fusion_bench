# FusionBench: A Comprehensive Benchmark of Deep Model Fusion

!!! note
    
    Stay tuned. Working in progress.

!!! note
    
    - Any questions or comments can be directed to the [GitHub Issues](https://github.com/tanganke/fusion_bench/issues) page for this project.
    - Any contributions or pull requests are welcome.


## Introduction to Deep Model Fusion

Deep model fusion is a technique that merges, ensemble, or fuse multiple deep neural networks to obtain a unified model.
It can be used to improve the performance and rubustness of model or to combine the strengths of different models, such as fuse multiple task-specific models to create a multi-task model.
For a more detailed introduction to deep model fusion, you can refer to [W. Li, 2023, 'Deep Model Fusion: A Survey'](http://arxiv.org/abs/2303.16203).
In this benchmark, we evaluate the performance of different fusion methods on a variety of datasets and tasks.

## General Structure

![alt text](images/framework_of_model_fusion.png)

FusionBench is a pioneering project that provides a comprehensive benchmark for deep model fusion, facilitating the evaluation and comparison of various model fusion techniques. The project is meticulously designed to support rigorous analysis and experimentation in the field of model fusion, offering a versatile and modular codebase tailored for advanced research and development.

The general structure of the FusionBench project can be visualized through its modular framework, which is divided into several key components:

1. **Analysis Tools**: This includes utilities for analyzing the Loss Landscape, Task Similarity, and Routing Analysis. These tools help researchers understand and interpret the underlying mechanics and performance metrics of different model fusion strategies.

2. **Model Pool**: A repository of various pre-trained models that can be accessed and utilized for fusion. This pool serves as the foundation for creating new, fused models by leveraging the strengths of each individual model.

3. **Fusion Algorithm**: The core component where Model Fusion takes place. It integrates models from the Model Pool and adjusts them according to the specified fusion algorithms. The output is then evaluated for performance and effectiveness.

4. **Task Pool**: A collection of tasks that the fused models are evaluated on. These tasks help in assessing the practical applicability and robustness of the fused models.

5. **YAML Configurations**: Central to the project's modularity, YAML files are used to configure models, datasets, and metrics, allowing seamless customization and scalability.

6. **Models & Warpers, Datasets, and Metrics**: These underlying modules include:
      - Models & Warpers: Tools and scripts for model loading, wrapping, and pre-processing.
      - Datasets: The datasets used for training, validation, and testing the fused models.
      - Metrics: The performance metrics used to evaluate the models, providing a comprehensive understanding of their capabilities.

By organizing these components into a structured and modular codebase, FusionBench ensures flexibility, ease of use, and scalability for researchers and developers. The project not only serves as a benchmark but also as a robust platform for innovation in the realm of deep model fusion.

## Getting Started

### Installation

```bash
# install from github repository
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

For detailed information on the options available, you can refer to this [page](/cli/fusion_bench/).