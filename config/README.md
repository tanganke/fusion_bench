# FusionBench Configuration

This directory contains configuration files for FusionBench. 
These configurations are essential for setting up and managing various algorithms and their hyperparameters.

## Built on Hydra

FusionBench's configuration system is built on [Hydra](https://hydra.cc/), a powerful framework for configuring complex applications. If you're new to Hydra, we recommend starting with the [Hydra documentation](https://hydra.cc/docs/intro/) to understand concepts like:

- Configuration composition and defaults
- Override syntax
- Configuration groups
- Variable interpolation

## Quick Start Example

Here's a minimal working configuration to run a fusion experiment:

**1. Create a config file** (e.g., `config/method/simple_example.yaml`):

```yaml
_target_: fusion_bench.method.simple_average.SimpleAverageAlgorithm
```

**2. Run with CLI:**

```bash
fusion_bench \
  method=simple_example \
  modelpool=your_modelpool_config \
  taskpool=your_taskpool_config
```

**3. Override parameters on the fly:**

```bash
fusion_bench \
  method=simple_example \
  method.hyperparam=value \
  modelpool=your_modelpool \
  taskpool=your_taskpool
```

## Configuration Structure

## Configuration Structure

FusionBench employs a modular configuration system, which is divided into three primary groups:

1. **Method Configuration**: Defines the fusion algorithm and its associated hyperparameters.
   - Location: `config/method/`
   - Each algorithm has its own directory with variant-specific configs

2. **Model Pool Configuration**: Manages the models involved in the fusion process, including datasets, tokenizers, preprocessors, and other related resources.
   - Location: `config/modelpool/`

3. **Task Pool Configuration**: Specifies the tasks and their corresponding datasets used for evaluating the fused models.
   - Location: `config/taskpool/`
