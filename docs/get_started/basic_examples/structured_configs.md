# Structured Config Groups

Due to the modular design of FusionBench, it employs a powerful structured configuration system based on [Hydra](https://hydra.cc/), which allows for modular, composable, and hierarchical configuration management. This document explains how to understand and use the structured config groups in FusionBench.

## Overview

The configuration system is organized into several key groups, each serving a specific purpose in the model fusion pipeline:

- **Method**: Defines the fusion algorithm and its hyperparameters
- **ModelPool**: Manages the models involved in the fusion process
- **TaskPool**: Specifies evaluation tasks and datasets
- **Hydra**: Controls Hydra framework settings
- **Fabric**: PyTorch Lightning Fabric configurations

## Configuration Directory Structure

```
config/
├── method/            # Fusion algorithms
├── modelpool/         # Model pool configurations
├── taskpool/          # Task pool configurations
├── hydra/             # Hydra framework settings
├── fabric/            # Fabric configurations
└── *.yaml             # Top-level configuration files
```

## Core Configuration Groups

### 1. Method Configuration (`method/`)

The method configuration defines which fusion algorithm to use and its hyperparameters. Each method configuration file specifies:

- `_target_`: The Python class implementing the algorithm
- Algorithm-specific parameters

**Example: Simple Average** (hyperparameter-free)
```yaml
_target_: fusion_bench.method.SimpleAverageAlgorithm
```

**Example: Task Arithmetic**
```yaml
_target_: fusion_bench.method.TaskArithmeticAlgorithm
scaling_factor: 0.3
```

### 2. ModelPool Configuration (`modelpool/`)

ModelPool configurations define the collection of models to be fused, including:

- Base/pretrained models
- Fine-tuned models for specific tasks
- Model preprocessors and processors

**Example: CLIP Vision Model Pool**
```yaml
_target_: fusion_bench.modelpool.CLIPVisionModelPool
_recursive_: False
processor: openai/clip-vit-base-patch32
models:
  _pretrained_: openai/clip-vit-base-patch32
  sun397: tanganke/clip-vit-base-patch32_sun397
  stanford-cars: tanganke/clip-vit-base-patch32_stanford-cars
platform: hf
```

### 3. TaskPool Configuration (`taskpool/`)

TaskPool configurations specify the evaluation tasks and their datasets:

- Test datasets for evaluation
- Task-specific processors

**Example: CLIP Vision Task Pool**
```yaml
_target_: fusion_bench.taskpool.CLIPVisionModelTaskPool
test_datasets:
  sun397:
    _target_: datasets.load_dataset
    path: tanganke/sun397
    split: test
  stanford-cars:
    _target_: datasets.load_dataset
    path: tanganke/stanford_cars
    split: test
clip_model: openai/clip-vit-base-patch32
processor: openai/clip-vit-base-patch32
```

### 5. Hydra Configuration (`hydra/`)

Controls Hydra framework behavior:

```yaml title="config/hydra/default.yaml"
--8<-- "config/hydra/default.yaml"
```

## Complete Configuration Example

Here's a complete configuration that combines all groups:

```yaml title="config/_get_started/clip_task_arithmetic.yaml" linenums="1" hl_lines="5"
--8<-- "config/_get_started/clip_task_arithmetic.yaml"
```

## Using Structured Configs

### 1. With Hydra Defaults

You can use Hydra's defaults system to compose configurations:

```yaml
defaults:
  - method: simple_average
  - modelpool: CLIPVisionModelPool/clip-vit-base-patch32_TA8
  - taskpool: CLIPVisionModelTaskPool/clip-vit-classification_TA8.yaml
  - _self_

_target_: fusion_bench.programs.FabricModelFusionProgram
```

### 2. Command Line Overrides

Override configuration values from the command line:

```bash
fusion_bench method=task_arithmetic method.scaling_factor=0.5
```

### 3. Config Groups

Specify different config groups:

```bash
fusion_bench \
  method=adamerging/clip \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-base-patch32_robustness_corrupted
```

## Advanced Features

### Selecting Multiple Configs from a Config Group

In some scenarios, you may need to select multiple configuration files from the same config group. This is particularly useful when you want to load multiple datasets, models, or evaluation tasks in a single run.

#### The Problem

Sometimes you need to compose configurations where a single component requires multiple sub-configurations. For example:

- Loading multiple training datasets at once
- Evaluating on multiple test datasets simultaneously
- Combining multiple model architectures in an ensemble

#### The Solution

Use a **list of config names** as the value of the config group in the Defaults List or via command line.

#### Basic Example

Let's understand this with a simplified example. Suppose we want to configure training datasets for a model:

**Config structure:**
```
config/
├── datasets_config.yaml
└── dataset/
    ├── sun397.yaml
    ├── stanford-cars.yaml
    └── resisc45.yaml
```

**Using defaults list** (`datasets_config.yaml`):
```yaml
defaults:
  - dataset:
      - sun397
      - stanford-cars
      - resisc45
```

**Output:** All three dataset configs will be loaded and merged into your configuration.

#### FusionBench Example

Here's a practical example from FusionBench where we load multiple training datasets:

```yaml
defaults:
  - /dataset/image_classification/train@train_datasets:
      - sun397
      - stanford-cars
      - resisc45

# The @train_datasets syntax specifies the package path where 
# these configs will be placed in the final configuration
```

**Result:** The configuration will include all three datasets under `train_datasets`:
```yaml
train_datasets:
  sun397:
    _target_: datasets.load_dataset
    path: tanganke/sun397
    split: train
  stanford-cars:
    _target_: datasets.load_dataset
    path: tanganke/stanford_cars
    split: train
  resisc45:
    _target_: datasets.load_dataset
    path: tanganke/resisc45
    split: train
```

#### Package Relocation

You can relocate where the configs are placed in the final configuration using the `@` syntax:

```yaml
defaults:
  - dataset@training.data:
      - sun397
      - stanford-cars
```

This places the datasets under `training.data` instead of the default `dataset`:

```yaml
training:
  data:
    sun397: ...
    stanford-cars: ...
```

### Recursive Configuration

Control recursive instantiation with `_recursive_`:

```yaml
_target_: fusion_bench.modelpool.CLIPVisionModelPool
_recursive_: False  # Prevent recursive instantiation
```
