# Customize Algorithm

This tutorial demonstrates how to implement a new model merging algorithm in FusionBench. You'll learn the essential components, interfaces, and best practices for creating custom algorithms that integrate seamlessly with the FusionBench framework.

## 🎯 Overview

FusionBench follows a modular design where algorithms inherit from [`BaseAlgorithm`][fusion_bench.method.BaseAlgorithm] and implement the required interfaces. This allows you to:

- Create custom merging strategies
- Leverage existing utilities and mixins
- Integrate with the configuration system
- Add profiling and monitoring capabilities

## 🏗️ Algorithm Structure

### Basic Template

Here's the minimal structure for a custom algorithm:

```python linenums="1" hl_lines="35-52"
import torch
from typing import Union, Dict
from torch import nn

from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool


class CustomMergingAlgorithm(BaseAlgorithm):
    """
    Custom algorithm implementation.
    
    This algorithm demonstrates how to implement a new merging strategy.
    """
    
    # Configuration mapping for YAML serialization
    _config_mapping = BaseAlgorithm._config_mapping | {
        "custom_param": "custom_param",
        "another_param": "another_param",
    }

    def __init__(self, custom_param: float = 0.5, another_param: bool = True):
        """
        Initialize the algorithm with custom parameters.
        
        Args:
            custom_param: Example parameter for your algorithm
            another_param: Another example parameter
        """
        super().__init__()
        self.custom_param = custom_param
        self.another_param = another_param
    
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> nn.Module:
        """
        Implement your custom merging logic here.
        
        Args:
            modelpool: Collection of models to merge
            
        Returns:
            Merged model
        """
        # Convert dict to BaseModelPool if needed
        if isinstance(modelpool, dict):
            modelpool = BaseModelPool(modelpool)
        
        # Your custom merging logic goes here
        models = modelpool.models()
        merged_model = ...
        
        return merged_model
```

### Using `@auto_register_config` (Recommended)

FusionBench provides the [`@auto_register_config`][fusion_bench.mixins.auto_register_config] decorator as a modern, boilerplate-free alternative to manually defining `_config_mapping`. It automatically registers all `__init__` parameters and sets them as instance attributes:

```python linenums="1" hl_lines="5 9"
from typing import Union, Dict
from torch import nn

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import auto_register_config
from fusion_bench.modelpool import BaseModelPool


@auto_register_config
class CustomMergingAlgorithm(BaseAlgorithm):
    """
    Custom algorithm using auto_register_config decorator.
    Parameters are automatically registered and set as instance attributes.
    """

    def __init__(self, custom_param: float = 0.5, another_param: bool = True):
        super().__init__()
        # No need to manually set self.custom_param = custom_param etc.
        # @auto_register_config handles this automatically
    
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> nn.Module:
        if isinstance(modelpool, dict):
            modelpool = BaseModelPool(modelpool)
        
        # Access parameters via self attributes (set automatically)
        print(f"custom_param={self.custom_param}, another_param={self.another_param}")
        
        merged_model = ...
        return merged_model
```

The `@auto_register_config` approach is used throughout the FusionBench codebase (e.g., [`TiesMergingAlgorithm`][fusion_bench.method.TiesMergingAlgorithm], [`TaskArithmeticAlgorithm`][fusion_bench.method.TaskArithmeticAlgorithm]) and is the **recommended pattern for new algorithms**.

**Comparison:**

| Approach | Boilerplate | Explicit mapping | Suitable for |
|----------|------------|-----------------|--------------|
| Manual `_config_mapping` | More | Yes | Complex custom mappings |
| `@auto_register_config` | Minimal | Automatic | Most new algorithms |

## 📁 File Organization

Organize your algorithm files following FusionBench conventions:

```text
fusion_bench/method/
├── your_algorithm/
│   ├── __init__.py
│   └── your_algorithm.py
└── __init__.py  # Register your algorithm here (optional)
```

### (Optional) Register in Main __init__.py

Add your algorithm to the main method registry:

```python title="fusion_bench/method/__init__.py"
_import_structure = {
    # ... existing algorithms ...
    "your_algorithm": ["YourAlgorithm"],
    # ... rest of the algorithms ...
}

if TYPE_CHECKING:
    from .your_algorithm import YourAlgorithm
```

## ⚙️ Configuration Integration

Integrate your algorithm with the FusionBench configuration system by creating a YAML configuration file:

```yaml title="config/method/your_algorithm.yaml"
_target_: fusion_bench.method.YourAlgorithm
custom_param: 0.5
another_param: true
```

## 🔌 Advanced Features

### Adding Profiling Support

Use the `SimpleProfilerMixin` for performance monitoring:

```python
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin

class YourAlgorithm(BaseAlgorithm, SimpleProfilerMixin):
    
    def run(self, modelpool):
        with self.profile("initialization"):
            # Initialization code
            pass
        
        with self.profile("model_loading"):
            # Model loading code
            pass
        
        with self.profile("parameter_merging"):
            # Merging logic
            pass
        
        # Print timing summary
        self.print_profile_summary()
```

### Adding Logging

Include proper logging for debugging and monitoring:

```python
import logging

log = logging.getLogger(__name__)

class YourAlgorithm(BaseAlgorithm):
    
    def run(self, modelpool):
        log.info(f"Starting merge with {len(modelpool.model_names)} models")
        log.debug(f"Model names: {modelpool.model_names}")
        
        # Algorithm logic...
        
        log.info("Merge completed successfully")
```

### Parameter Validation

Add robust parameter validation:

```python
class YourAlgorithm(BaseAlgorithm):
    
    def __init__(self, param1: float, param2: int):
        super().__init__()
        
        # Validate parameters
        if not 0 <= param1 <= 1:
            raise ValueError(f"param1 must be in [0, 1], got {param1}")
        
        if param2 <= 0:
            raise ValueError(f"param2 must be positive, got {param2}")
        
        self.param1 = param1
        self.param2 = param2
```

## 🚀 Usage Examples

### Command Line Usage

```bash
# Use your custom algorithm
fusion_bench \
    method=<path_to_your_algorithm_config> \
    modelpool=modelpool_config \
    taskpool=taskpool_config
```

### Programmatic Usage (Use without CLI)

```python
from fusion_bench.method.your_algorithm import YourAlgorithm
from fusion_bench.modelpool import BaseModelPool

# Create your algorithm
algorithm = YourAlgorithm(
    custom_param=0.5,
    another_param=True
)

# Apply to your models
merged_model = algorithm.run(your_modelpool)
```

## 🔗 Next Steps

- Explore [existing algorithms](../../api/fusion_bench.method/index.md) for more implementation patterns
- Learn about [model pools](../../api/fusion_bench.modelpool.md) for advanced model management
- Check out [evaluation strategies](../../api/fusion_bench.taskpool.md) to assess your merged models
- Contribute your algorithm back to the FusionBench community!
