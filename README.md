# FusionBench: A Comprehensive Benchmark of Deep Model Fusion

> This project is still in testing phase as the API may be subject to change. Please report any issues you encounter.

Documentation is available at [tanganke.github.io/fusion_bench/](https://tanganke.github.io/fusion_bench/).

## Installation

```bash
# install from github repository
git clone https://github.com/tanganke/fusion_bench.git
cd fusion_bench

pip install -e . # install the package in editable mode
```

## Introduction to Deep Model Fusion

Deep model fusion is a technique that merges, ensemble, or fuse multiple deep neural networks to obtain a unified model.
It can be used to improve the performance and rubustness of model or to combine the strengths of different models, such as fuse multiple task-specific models to create a multi-task model.
For a more detailed introduction to deep model fusion, you can refer to [W. Li, 2023, 'Deep Model Fusion: A Survey'](https://arxiv.org/abs/2309.15698). We also provide a brief overview of deep model fusion in [our documentation](https://tanganke.github.io/fusion_bench/).
In this benchmark, we evaluate the performance of different fusion methods on a variety of datasets and tasks.

## Citation

If you find this benchmark useful, please consider citing our work:

```bibtex
@misc{tangFusionBenchComprehensiveBenchmark2024,
  title = {{{FusionBench}}: {{A Comprehensive Benchmark}} of {{Deep Model Fusion}}},
  shorttitle = {{{FusionBench}}},
  author = {Tang, Anke and Shen, Li and Luo, Yong and Hu, Han and Do, Bo and Tao, Dacheng},
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