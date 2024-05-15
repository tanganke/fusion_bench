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

```
fusion_bench [--config-path CONFIG_PATH] [--config-name CONFIG_NAME] \
    OPTION_1=VALUE_1 OPTION_2=VALUE_2 ...
```

