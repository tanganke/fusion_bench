---
title: Overview
---
# API Reference

??? bug "Breaking Changes in v0.2"

    Recent upgrade to version >= v0.2.0 may cause some breaking changes. Make some documented instructions may be outdated.
    You can install a specific version by `pip install fusion-bench==0.1.6` or checkout to a specific version by `git checkout v0.1.6`.
    If you encounter any issues, please feel free to raise an issue.
    We are working on the documentation and will update it as soon as possible. 
    
    **Use version >=0.2.0 is recommended.**

Here we provides an overview of the API reference for FusionBench.

## Entry Points

- [fusion_bench.scripts.cli.main][]

::: fusion_bench.scripts.cli.main

## Class Definitions

Base class for all FusionBench components.

- [fusion_bench.BaseAlgorithm][]: Base class for all algorithms.
- [fusion_bench.BaseModelPool][]: Base class for all model pools.
- [fusion_bench.BaseTaskPool][]: Base class for all task pools.

## Modules

- [fusion_bench.mixins](fusion_bench.mixins.md): Mixins.
- [fusion_bench.program](fusion_bench.program.md): Program definitions.
- [fusion_bench.method](fusion_bench.method/index.md): Implementation of methods.
- [fusion_bench.modelpool](fusion_bench.modelpool.md): Model pools.
- [fusion_bench.taskpool](fusion_bench.taskpool.md): Task pools.
- [fusion_bench.utils](fusion_bench.utils/index.md): Utility functions.
- [fusion_bench.models](fusion_bench.models.md): Model definitions and utilities.
- [fusion_bench.dataset](fusion_bench.dataset.md): Dataset definitions and utilities.
- [fusion_bench.tasks](fusion_bench.tasks.md): Task definitions and utilities.
- [fusion_bench.metrics](fusion_bench.metrics.md): Metric definitions and utilities.
- [fusion_bench.optim](fusion_bench.optim.md): Implementation of optimizers and learning rate schedulers.
- [fusion_bench.constants](fusion_bench.constants.md): Constant definitions.
- [fusion_bench.compat](fusion_bench.compat.md) (deprecated): Compatibility layer for v0.1.x, this is deprecated and will be removed in future versions.
