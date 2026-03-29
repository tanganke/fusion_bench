# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FusionBench is a comprehensive benchmark and toolkit for deep model fusion—merging, ensembling, or mixing multiple neural networks into unified models. It supports various fusion techniques across computer vision (CLIP, ResNet, OpenCLIP, DINOv2) and NLP (GPT-2, Llama, Flan-T5, Gemma) tasks.

## Documentation Structure

- **docs/get_started/**: Basic to advanced usage examples
- **docs/algorithms/**: Documentation for each fusion algorithm
- **docs/api/**: Auto-generated API reference (mkdocstrings)
- **docs/guides/**: Tutorials for specific use cases
- **docs/modelpool/**: Model pool documentation (including LLM guides)
- **docs/taskpool/**: Task pool documentation
- **docs/cli/**: CLI reference

## Core Architecture: The Three-Component System

FusionBench follows a strict three-component architecture:

### 1. **Method** (`fusion_bench/method/`)
The fusion algorithm implementation. Each algorithm inherits from `BaseAlgorithm`.

**Key Conventions:**
- File path: `fusion_bench/method/{method_name}/{variant}.py`
- Must implement `run(self, modelpool)` abstract method
- Use `_config_mapping` dict to map attributes to YAML config keys (or use `@auto_register_config` decorator)
- Lifecycle hooks: `on_run_start()` and `on_run_end()`
- Examples: `simple_average`, `task_arithmetic`, `adamerging`, `regmean`, `dare`, `ties_merging`, `model_stock`, `slerp`

### 2. **ModelPool** (`fusion_bench/modelpool/`)
Manages models and their datasets. Responsible for lazy-loading models from HuggingFace, local paths, or model definitions.

**Key Conventions:**
- Inherits from `BaseModelPool`
- Supports special model names: `_pretrained_` (base/pretrained model)
- Configuration specifies models as dict with model names as keys
- Can include `train_datasets`, `val_datasets`, `test_datasets` in config

### 3. **TaskPool** (`fusion_bench/taskpool/`)
Evaluates fused models on specific tasks/datasets.

**Key Conventions:**
- Inherits from `BaseTaskPool`
- Implements `evaluate(model)` to return metrics dict
- Test datasets configured in YAML under `test_datasets`

## Configuration System: Hydra-Based YAML

FusionBench uses [Hydra](https://hydra.cc/) for hierarchical configuration.

### Main Entry Point
- Default config: `config/fabric_model_fusion.yaml`
- Entry point: `fusion_bench/scripts/cli.py` (decorated with `@hydra.main`)

### Configuration Structure
```
config/
├── fabric_model_fusion.yaml     # Main entry point
├── method/                      # Fusion algorithms
│   ├── simple_average.yaml
│   └── task_arithmetic.yaml
├── modelpool/                   # Model pool definitions
│   └── CLIPVisionModelPool/
│       └── clip-vit-base-patch32_TA8.yaml
├── taskpool/                    # Task definitions
│   └── clip_vision.yaml
├── fabric/                      # Fabric configs
│   └── auto.yaml
└── path/                        # Path configs
    └── default.yaml
```

### YAML Configuration Pattern
Every component config MUST include `_target_` pointing to the Python class:

```yaml
# config/method/simple_average.yaml
_target_: fusion_bench.method.SimpleAverageAlgorithm
# algorithm hyperparameters here
```

### `_config_mapping` Pattern
To serialize class attributes to YAML config:

**Option 1: Manual mapping with `_config_mapping`:**
```python
class MyAlgorithm(BaseAlgorithm):
    _config_mapping = BaseAlgorithm._config_mapping | {
        "learning_rate": "lr",
        "num_epochs": "epochs",
    }
```

**Option 2: Automatic registration with `@auto_register_config` decorator:**
```python
from fusion_bench.mixins import auto_register_config

@auto_register_config
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, learning_rate: float, num_epochs: int):
        super().__init__()
```

## Running FusionBench

### CLI Usage (Primary Interface)
```bash
# Basic command structure
fusion_bench \
  method=simple_average \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=clip_vision \
  method.some_param=value  # Override config values

# Dry run (show configuration without executing)
fusion_bench --dry_run

# Quick testing (single batch)
fusion_bench --fast_dev_run
```

### WebUI
```bash
fusion_bench_webui  # Launch Gradio-based command generator
```

### Important CLI Options
- `--dry_run`: Validate configuration without executing
- `--fast_dev_run`: Quick testing with single batch
- `--cfg, -c`: Show config instead of running
- `--print_config`: Print configuration with rich formatting
- `merged_model_save_path`: Save merged model to specified path
- `report_save_path`: Save evaluation report as JSON

### Debugging in VSCode
Configure `.vscode/launch.json` with:
```json
{
    "name": "FusionBench",
    "type": "debugpy",
    "request": "launch",
    "module": "fusion_bench.scripts.cli",
    "args": "${command:pickArgs}",
    "console": "integratedTerminal"
}
```

### Programmatic Usage
```python
from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils import instantiate

# Load algorithm from config
algorithm = BaseAlgorithm.from_yaml("config/method/simple_average.yaml")

# Run fusion
merged_model = algorithm.run(modelpool)
```

## Critical Development Patterns

### 1. Lazy Import System
FusionBench uses custom `LazyImporter` to avoid importing heavy dependencies (PyTorch, Transformers) until needed.

**Pattern in `__init__.py`:**
```python
from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    "module_name": ["ClassName1", "ClassName2"],
}

if TYPE_CHECKING:
    from .module_name import ClassName1, ClassName2
else:
    sys.modules[__name__] = LazyImporter(__name__, ...)
```

### 2. Naming Conventions
- **Methods**: `fusion_bench/method/{method_name}/{variant}.py`
- **Config**: `config/method/{method_name}/{variant}.yaml`
- **Tests**: `tests/test_{feature}.py`
- **Examples**: `examples/{method_name}/`

### 3. Lightning Fabric Integration
For distributed training/inference, FusionBench uses PyTorch Lightning Fabric.

```python
from fusion_bench.mixins import LightningFabricMixin

class MyAlgorithm(LightningFabricMixin, BaseAlgorithm):
    def run(self, modelpool):
        # Access fabric via self.fabric
        model = self.fabric.setup_module(model)
```

### 4. Mixins Architecture
Common mixins in `fusion_bench/mixins/`:
- `BaseYAMLSerializable` / `YAMLSerializationMixin`: YAML serialization
- `HydraConfigMixin`: Hydra integration
- `LightningFabricMixin`: Fabric device management
- `SimpleProfilerMixin`: Performance profiling
- `PyinstrumentProfilerMixin`: Pyinstrument profiling
- `CLIPClassificationMixin`: CLIP-specific evaluation helpers

**Mixin Order Matters:** Put functionality mixins before base class:
```python
class MyClass(MixinA, MixinB, BaseClass):
    pass
```

## Common Commands

### Run Tests
```bash
# Run all tests
python -m unittest discover -v -s ./tests -p "test_*.py"

# Run specific test
python -m unittest tests.test_simple_average
```

### Offline Mode
To run in offline mode (using only cached models):
```bash
source offline_mode.sh
```

### Documentation
```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve docs locally
mkdocs serve
```

### Code Formatting
```bash
# Format Python code
black .
isort .

# Format YAML (requires yamlfmt)
yamlfmt config/
```

## Key Files to Reference

- **CLI entry**: `fusion_bench/scripts/cli.py`
- **Main programs**:
  - `fusion_bench/programs/fabric_fusion_program.py` - `FabricModelFusionProgram` (recommended)
  - `fusion_bench/programs/fusion_program.py` - `ModelFusionProgram` (non-fabric version)
  - `fusion_bench/programs/base_program.py` - `BaseHydraProgram`
- **Base classes**:
  - `fusion_bench/method/base_algorithm.py` - BaseAlgorithm
  - `fusion_bench/modelpool/base_pool.py` - BaseModelPool
  - `fusion_bench/taskpool/base_pool.py` - BaseTaskPool
  - `fusion_bench/programs/base_program.py` - BaseHydraProgram
- **Utils**:
  - `fusion_bench/utils/instantiate_utils.py` - Custom instantiation logic
  - `fusion_bench/utils/lazy_imports.py` - LazyImporter for deferred imports
  - `fusion_bench/utils/hydra_utils.py` - Hydra helper functions
- **Mixins**:
  - `fusion_bench/mixins/hydra_config.py` - Hydra integration, auto_register_config
  - `fusion_bench/mixins/lightning_fabric.py` - Lightning Fabric integration
  - `fusion_bench/mixins/serialization.py` - YAML serialization
- **Main config**: `config/fabric_model_fusion.yaml` (default entry point)

## Implementing New Features

### New Fusion Algorithm
1. Create `fusion_bench/method/{name}/{variant}.py` inheriting `BaseAlgorithm`
2. Implement `run(self, modelpool)` method
3. Create matching `config/method/{name}/{variant}.yaml` with `_target_`
4. Add to `fusion_bench/method/__init__.py` import structure

### New Model Type
1. Create `fusion_bench/modelpool/{name}.py` inheriting `BaseModelPool`
2. Implement `load_model(self, model_name)` method
3. Create config in `config/modelpool/{name}/`

### New Evaluation Task
1. Create `fusion_bench/taskpool/{name}.py` inheriting `BaseTaskPool`
2. Implement `evaluate(self, model)` method
3. Create config in `config/taskpool/{name}.yaml`

## Common Pitfalls to Avoid

1. **Don't bypass the three-component system**: Always use ModelPool, not direct model loading in methods
2. **Don't forget `_target_` in configs**: Every instantiated config needs this
3. **Don't ignore `_config_mapping`**: Required for proper YAML serialization (or use `@auto_register_config`)
4. **Don't import PyTorch/Transformers at module level**: Use lazy imports via `LazyImporter`
5. **Don't hardcode paths**: Use Hydra's path configuration system
6. **Don't mix positional and keyword arguments carelessly**: When using `@auto_register_config`, be aware of how arguments are processed
