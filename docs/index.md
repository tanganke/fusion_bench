---
title: Welcome to FusionBench
description: A Comprehensive Benchmark/Toolkit of Deep Model Fusion
---

# **WELCOME TO 🧬 FUSIONBENCH**

[![arXiv](https://img.shields.io/badge/arXiv-2406.03280-b31b1b.svg)](http://arxiv.org/abs/2406.03280)
[![GitHub License](https://img.shields.io/github/license/tanganke/fusion_bench)](https://github.com/tanganke/fusion_bench/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/fusion-bench)](https://pypi.org/project/fusion-bench/)
[![Downloads](https://static.pepy.tech/badge/fusion-bench/month)](https://pepy.tech/project/fusion-bench)
[![Static Badge](https://img.shields.io/badge/doc-mkdocs-blue)](https://tanganke.github.io/fusion_bench/)
[![Static Badge](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Static Badge](https://img.shields.io/badge/code%20style-yamlfmt-black)](https://github.com/google/yamlfmt)

## Install FusionBench

Prerequisites:

- Python 3.10 or later (some features may not work as expected for earlier versions)

=== "Install from GitHub repository"

    Install the latest version of `fusion-bench` from GitHub repository and install it in editable mode by passing the `-e` flag to pip.

    ```bash
    git clone https://github.com/tanganke/fusion_bench.git
    cd fusion_bench

    # checkout to use a specific version. for example, v0.1.6
    # git checkout v0.1.6

    pip install -e . # install the package in editable mode
    ```

=== "Install from PyPi"

    `fusion-bench` can also be installed from PyPI as a library and toolkit for deep model fusion.

    ```bash
    pip install fusion-bench

    # you can also install a specific version
    # pip install fusion-bench==0.1.6
    ```

Installing `fusion-bench` will also install the latest [stable PyTorch](https://pytorch.org/) if you don't have it already.

## Next Steps

<div class="grid cards" markdown>

- **Learn More about FusionBench**

    ---

    Learn the basic concepts of FusionBench and the command line interface (CLI) as well as the programmatic usage of FusionBench.

    [:octicons-arrow-right-24: Read More](get_started/index.md)

- **Learn More About Deep Model Fusion**

    ---

    Read an introduction to deep model fusion and learn about its key concepts, techniques, and applications.

    [:octicons-arrow-right-24: Read More](introduction_to_model_fusion.md)

</div>


!!! note "Contributing to FusionBench"

    - Any questions or comments can be directed to the [GitHub Issues](https://github.com/tanganke/fusion_bench/issues) page for this project.
    - Any contributions or pull requests are welcome. If you find any mistakes or have suggestions for improvements, please feel free to raise an issue or submit a pull request.

!!! tip "Introduction to Deep Model Fusion (The Learn From Model Paradigm)"

    Deep model fusion is a technique that merges, ensemble, or fuse multiple deep neural networks to obtain a unified model.
    It can be used to improve the performance and robustness of model or to combine the strengths of different models, such as fuse multiple task-specific models to create a multi-task model.
    For a more detailed introduction to deep model fusion, you can refer to [W. Li, 2023, 'Deep Model Fusion: A Survey'](https://arxiv.org/abs/2309.15698). 
    In this benchmark, we evaluate the performance of different fusion methods on a variety of datasets and tasks. ...

    [:octicons-arrow-right-24: Read More](introduction_to_model_fusion.md)

## Citation

If you find this benchmark useful, please consider citing our work:

```bibtex
@article{tang2024fusionbench,
  title={Fusionbench: A comprehensive benchmark of deep model fusion},
  author={Tang, Anke and Shen, Li and Luo, Yong and Hu, Han and Du, Bo and Tao, Dacheng},
  journal={arXiv preprint arXiv:2406.03280},
  year={2024}
}
```
