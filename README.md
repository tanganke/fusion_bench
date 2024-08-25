# FusionBench: A Comprehensive Benchmark of Deep Model Fusion

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](http://arxiv.org/abs/2406.03280)
[![GitHub License](https://img.shields.io/github/license/tanganke/fusion_bench)](https://github.com/tanganke/fusion_bench/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/fusion-bench)](https://pypi.org/project/fusion-bench/)
[![Downloads](https://static.pepy.tech/badge/fusion-bench/month)](https://pepy.tech/project/fusion-bench)
[![Static Badge](https://img.shields.io/badge/doc-mkdocs-blue)](https://tanganke.github.io/fusion_bench/)
[![Static Badge](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)


> [!WARNING]  
> This project is still in testing phase as the API may be subject to change. Please report any issues you encounter.

> [!TIP]  
> Documentation is available at [tanganke.github.io/fusion_bench/](https://tanganke.github.io/fusion_bench/).


## Overview

FusionBench is a benchmark suite designed to evaluate the performance of various deep model fusion techniques. It aims to provide a comprehensive comparison of different methods on a variety of datasets and tasks.

## Installation

install from PyPI:

```bash
pip install fusion-bench
```

or install the latest version in development from github repository

```bash
git clone https://github.com/tanganke/fusion_bench.git
cd fusion_bench

pip install -e . # install the package in editable mode
```

## Introduction to Deep Model Fusion

Deep model fusion is a technique that merges, ensemble, or fuse multiple deep neural networks to obtain a unified model.
It can be used to improve the performance and robustness of model or to combine the strengths of different models, such as fuse multiple task-specific models to create a multi-task model.
For a more detailed introduction to deep model fusion, you can refer to [W. Li, 2023, 'Deep Model Fusion: A Survey'](https://arxiv.org/abs/2309.15698). We also provide a brief overview of deep model fusion in [our documentation](https://tanganke.github.io/fusion_bench/).
In this benchmark, we evaluate the performance of different fusion methods on a variety of datasets and tasks.

## Project Structure

The project is structured as follows:

- `fusion_bench/`: the main package of the benchmark.
- `config/`: configuration files for the benchmark. We use [Hydra](https://hydra.cc/) to manage the configurations.
- `docs/`: documentation for the benchmark. We use [mkdocs](https://www.mkdocs.org/) to generate the documentation. Start the documentation server locally with `mkdocs serve`. The required packages can be installed with `pip install -r mkdocs-requirements.txt`.
- `examples/`: example scripts for running some of the experiments.
- `tests/`: unit tests for the benchmark.

## A Unified Command Line Interface

The `fusion_bench` command-line interface is a powerful tool for researchers and practitioners in the field of model fusion. It provides a streamlined way to experiment with various fusion algorithms, model combinations, and evaluation tasks. 
By leveraging Hydra's configuration management, fusion_bench offers flexibility in setting up experiments and reproducibility in results. 
The CLI's design allows for easy extension to new fusion methods, model types, and tasks, making it a versatile platform for advancing research in model fusion techniques.

Read the [CLI documentation](https://tanganke.github.io/fusion_bench/cli/fusion_bench/) for more information.

### FusionBench Command Generator WebUI

FusionBench Command Generator is a user-friendly web interface for generating FusionBench commands based on configuration files. 
It provides an interactive way to select and customize FusionBench configurations, making it easier to run experiments with different settings.
[Read more here](https://tanganke.github.io/fusion_bench/cli/fusion_bench_webui/).

![FusionBench Command Generator Web Interface](docs/cli/images/fusion_bench_webui.png)

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
