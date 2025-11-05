# FusionBench AI Coding Agent Instructions

## Project Overview

FusionBench is a comprehensive benchmark and toolkit for deep model fusion—merging, ensembling, or mixing multiple neural networks into unified models. The project supports various fusion techniques across computer vision (CLIP, ResNet, etc.) and NLP (GPT-2, Llama, Flan-T5) tasks.

**Core Purpose**: Enable researchers to experiment with model fusion algorithms through a unified, configuration-driven CLI and programmatic API.

## Architecture: The Three-Component System

FusionBench follows a strict three-component architecture that you MUST understand:

### 1. **Method** (`fusion_bench/method/`)
The fusion algorithm implementation. Each algorithm inherits from `BaseAlgorithm` (or `BaseModelFusionAlgorithm`).

```python
# Example: fusion_bench/method/simple_average/simple_average.py
class SimpleAverageAlgorithm(BaseAlgorithm):
    def run(self, modelpool: BaseModelPool):
        # Implement fusion logic here
        return merged_model
```

**Key Conventions**:
- File path: `fusion_bench/method/{method_name}/{variant}.py`
- Must implement `run(self, modelpool)` abstract method
- Use `_config_mapping` dict to map attributes to YAML config keys
- Lifecycle hooks: `on_run_start()` and `on_run_end()`
- Algorithm examples: `simple_average`, `task_arithmetic`, `adamerging`, `regmean`

### 2. **ModelPool** (`fusion_bench/modelpool/`)
Manages models and their datasets. Responsible for lazy-loading models from HuggingFace, local paths, or model definitions.

```python
# Example: fusion_bench/modelpool/clip_vision.py
class CLIPVisionModelPool(BaseModelPool):
    def load_model(self, model_name: str) -> nn.Module:
        # Load and return model
```

**Key Conventions**:
- Inherits from `BaseModelPool`
- Supports special model names: `_pretrained_` (base/pretrained model)
- Configuration specifies models as dict with model names as keys
- Can include `train_datasets`, `val_datasets`, `test_datasets` in config
- Property `model_names` returns only regular models (excludes special ones like `_pretrained_`)

### 3. **TaskPool** (`fusion_bench/taskpool/`)
Evaluates fused models on specific tasks/datasets.

```python
# Example: fusion_bench/taskpool/clip_vision.py
class CLIPVisionModelTaskPool(BaseTaskPool):
    def evaluate(self, model: nn.Module) -> Dict[str, Any]:
        # Run evaluation and return metrics
```

**Key Conventions**:
- Inherits from `BaseTaskPool`
- Implements `evaluate(model)` to return metrics dict
- Test datasets configured in YAML under `test_datasets`

## Configuration System: Hydra-Based YAML

FusionBench uses [Hydra](https://hydra.cc/) for hierarchical configuration. Understanding this is CRITICAL.

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
└── taskpool/                    # Task definitions
    └── clip_vision.yaml
```

### YAML Configuration Pattern
Every component config MUST include `_target_` pointing to the Python class:

```yaml
# config/method/simple_average.yaml
_target_: fusion_bench.method.SimpleAverageAlgorithm
# algorithm hyperparameters here

# config/modelpool/my_pool.yaml
_target_: fusion_bench.modelpool.CLIPVisionModelPool
models:
  _pretrained_: openai/clip-vit-base-patch32
  task1: path/to/finetuned/model
  task2: path/to/another/model
```

### `_config_mapping` Pattern
To serialize class attributes to YAML config, define `_config_mapping`:

```python
class MyAlgorithm(BaseAlgorithm):
    _config_mapping = BaseAlgorithm._config_mapping | {
        "learning_rate": "lr",      # self.learning_rate -> config.lr
        "num_epochs": "epochs",     # self.num_epochs -> config.epochs
    }
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

# The CLI entry point is fusion_bench/scripts/cli.py
# It uses @hydra.main decorator with config_path pointing to config/
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

**Pattern in `__init__.py`**:
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

**DO NOT** import heavy modules at module level in implementations.

### 2. Naming Conventions

- **Methods**: `fusion_bench/method/{method_name}/{variant}.py`
  - Example: `fusion_bench/method/regmean/clip_regmean.py`
- **Config**: `config/method/{method_name}/{variant}.yaml`
  - Example: `config/method/regmean/clip_regmean.yaml`
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
        # Use fabric for device management
```

Configuration in YAML:
```yaml
# config/fabric/auto.yaml defines fabric settings
# Main config references: fabric: auto
```

### 4. Mixins Architecture

FusionBench uses mixins for cross-cutting concerns. Common mixins in `fusion_bench/mixins/`:

- `BaseYAMLSerializable`: YAML serialization support
- `HydraConfigMixin`: Hydra integration
- `LightningFabricMixin`: Fabric device management
- `SimpleProfilerMixin`: Performance profiling
- `CLIPClassificationMixin`: CLIP-specific evaluation helpers

**Mixin Order Matters**: Put functionality mixins before base class:
```python
class MyClass(MixinA, MixinB, BaseClass):
    pass
```

## Testing

```bash
# Run all tests
python -m unittest discover -v -s ./tests -p "test_*.py"

# Run specific test
python -m unittest tests.test_simple_average
```

Tests should:
1. Use small models/configs for speed
2. Test the three-component interaction (method + modelpool + taskpool)
3. Validate YAML config loading/saving

## Documentation

- **MkDocs** for documentation: `mkdocs serve` to preview locally
- Install docs dependencies: `pip install -e ".[docs]"`
- Docs in `docs/`, structured to match code organization
- API references auto-generated from docstrings using `mkdocstrings`

## Common Pitfalls to Avoid

1. **Don't bypass the three-component system**: Always use ModelPool, not direct model loading in methods
2. **Don't forget `_target_` in configs**: Every instantiated config needs this
3. **Don't use absolute imports in examples**: Examples should be self-contained
4. **Don't ignore `_config_mapping`**: Required for proper YAML serialization
5. **Don't import PyTorch/Transformers at module level**: Use lazy imports
6. **Don't hardcode paths**: Use Hydra's path configuration system (`config/path/default.yaml`)

## Key Files to Reference

- **Base classes**: `fusion_bench/method/base_algorithm.py`, `fusion_bench/modelpool/base_pool.py`, `fusion_bench/taskpool/base_pool.py`
- **CLI entry**: `fusion_bench/scripts/cli.py`
- **Main program**: `fusion_bench/programs/fabric_fusion_program.py`
- **Utils**: `fusion_bench/utils/instantiate_utils.py` (custom instantiation logic)
- **Example config**: `config/_get_started/clip_simple_average.yaml`

## Quick Start for New Features

1. **New fusion algorithm**:
   - Create `fusion_bench/method/{name}/{variant}.py` inheriting `BaseAlgorithm`
   - Implement `run(self, modelpool)` method
   - Create matching `config/method/{name}/{variant}.yaml` with `_target_`
   - Add to `fusion_bench/method/__init__.py` import structure

2. **New model type**:
   - Create `fusion_bench/modelpool/{name}.py` inheriting `BaseModelPool`
   - Implement `load_model(self, model_name)` method
   - Create config in `config/modelpool/{name}/`

3. **New evaluation task**:
   - Create `fusion_bench/taskpool/{name}.py` inheriting `BaseTaskPool`
   - Implement `evaluate(self, model)` method
   - Create config in `config/taskpool/{name}.yaml`

## Code Style

- **Black** for formatting: `black .`
- **isort** for imports: `isort .`
- Python 3.10+ required (uses pattern matching in some places)
- Comprehensive docstrings (Google style) for all public APIs
