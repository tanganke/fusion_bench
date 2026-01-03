# FusionBench Configuration

This directory contains configuration files for FusionBench. 
These configurations are essential for setting up and managing various algorithms and their hyperparameters.

## Built on Hydra

FusionBench's configuration system is built on [Hydra](https://hydra.cc/), a powerful framework for configuring complex applications. If you're new to Hydra, we recommend starting with the [Hydra documentation](https://hydra.cc/docs/intro/) to understand concepts like:

- Configuration composition and defaults
- Override syntax
- Configuration groups
- Variable interpolation

## Configuration Structure

FusionBench employs a modular configuration system, which is divided into three primary groups:

1. **Method Configuration**: Defines the fusion algorithm and its associated hyperparameters.
2. **Model Pool Configuration**: Manages the models involved in the fusion process, including datasets, tokenizers, preprocessors, and other related resources.
3. **Task Pool Configuration**: Specifies the tasks and their corresponding datasets used for evaluating the fused models.
