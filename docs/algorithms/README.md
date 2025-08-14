# Algorithm Module

The Algorithm module is a core component of FusionBench, dedicated to the implementation and execution of various model fusion techniques. This module provides the mechanisms necessary to combine multiple models from model pools, enabling sophisticated and optimized model merging operations.

## Algorithm Configuration Structure

Algorithms use Hydra-based configuration with the `_target_` field:

### Basic Algorithm Configuration

```yaml
# Simple algorithm with no parameters
_target_: fusion_bench.method.SimpleAverageAlgorithm
```

### Parameterized Algorithm Configuration

```yaml
# Algorithm with parameters
_target_: fusion_bench.method.TaskArithmeticAlgorithm
scaling_factor: 0.3
```

### Advanced Algorithm Configuration

```yaml
# Complex algorithm with multiple parameters
_target_: fusion_bench.method.MoreAdvancedAlgorithm
weights_initial: [0.3, 0.3, 0.4]  
layer_wise_weight: false
entropy_k: 1
entropy_regularization_weight: 0.001
test_time_adaptation_steps: 100
```

## Implementation Architecture

All fusion algorithms inherit from [`BaseAlgorithm`][fusion_bench.BaseAlgorithm]:

```python
from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool

class CustomAlgorithm(BaseAlgorithm):
    """
    Custom model fusion algorithm implementation.
    """
    
    # Configuration mapping for YAML serialization
    _config_mapping = BaseAlgorithm._config_mapping | {
        "custom_param": "custom_param",
        "another_param": "another_param",
    }

    def __init__(self, custom_param: float = 0.5, another_param: bool = True, **kwargs):
        """Initialize the algorithm with custom parameters."""
        self.custom_param = custom_param
        self.another_param = another_param
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        """
        Execute the fusion algorithm.
        
        Args:
            modelpool: Pool of models to fuse
            
        Returns:
            Fused model (torch.nn.Module)
        """
        # Your fusion logic here
        pretrained_model = modelpool.load_pretrained_model()
        models = [modelpool.load_model(name) for name in modelpool.model_names]
        
        # Implement your fusion strategy
        merged_model = self.merge_models(pretrained_model, models)
        return merged_model
```

## Usage Examples

### Direct Instantiation

```python
from fusion_bench.method import TaskArithmeticAlgorithm
from fusion_bench.modelpool import BaseModelPool

# Create algorithm directly
algorithm = TaskArithmeticAlgorithm(scaling_factor=0.3)

# Apply to your model pool
merged_model = algorithm.run(your_modelpool)
```

### Configuration-Based Usage

```python
from fusion_bench.utils import instantiate
from omegaconf import OmegaConf

# Load from configuration
config = OmegaConf.load("config/method/task_arithmetic.yaml")
algorithm = instantiate(config)

# Execute fusion
merged_model = algorithm.run(modelpool)
```

### Integration with Programs

The most common usage is through FusionBench programs:

```python
from fusion_bench.programs import FabricModelFusionProgram

# Full workflow using program
program = FabricModelFusionProgram(
    method=method_config,
    modelpool=modelpool_config, 
    taskpool=taskpool_config
)

# This runs: algorithm.run(modelpool) + evaluation
program.run()
```

### Command Line Usage

```bash
fusion_bench \
    method=task_arithmetic \
    method.scaling_factor=0.3 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

## Advanced Features

### Profiling Support

Many algorithms support profiling through `SimpleProfilerMixin`:

```python
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin

class ProfilingAlgorithm(BaseAlgorithm, SimpleProfilerMixin):
    
    def run(self, modelpool):
        with self.profile("initialization"):
            # Initialization code
            pass
        
        with self.profile("model_merging"):
            # Merging logic
            pass
        
        # Print timing summary
        self.print_profile_summary()
```

### Lightning Fabric Integration

For distributed and accelerated computing:

```python
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import LightningFabricMixin

class DistributedAlgorithm(BaseAlgorithm, LightningFabricMixin):
    
    def run(self, modelpool):
        # Access fabric for distributed operations
        if hasattr(self, 'fabric'):
            # Use self.fabric for distributed operations
            pass
        
        # Algorithm implementation
        merged_model = self.merge_models(modelpool)
        return merged_model
```

### Integration with TaskPools

Algorithms can access taskpools for evaluation during fusion:

```python
class AdaptiveAlgorithm(BaseAlgorithm):
    
    def run(self, modelpool):
        # Access taskpool if available through program
        if hasattr(self, '_program') and self._program.taskpool:
            # Use taskpool for adaptive fusion
            taskpool = self._program.taskpool
            
            for step in range(self.adaptation_steps):
                merged_model = self.merge_step(modelpool)
                results = taskpool.evaluate(merged_model)
                self.update_weights(results)
        
        return merged_model
```

## Migration from v0.1.x

If you're migrating from v0.1.x, note these key changes:

1. **Base Class**: Use [`BaseAlgorithm`][fusion_bench.BaseAlgorithm] instead of [`ModelFusionAlgorithm`][fusion_bench.compat.method.ModelFusionAlgorithm]
2. **Configuration**: Use `_target_` fields instead of string-based algorithm names  
3. **Instantiation**: Use `instantiate(config)` instead of factory methods
4. **Parameters**: Pass parameters to `__init__` instead of through config dict

### Migration Example

```python
# Old (v0.1.x, deprecated)
from fusion_bench.compat.method import ModelFusionAlgorithm, AlgorithmFactory

class OldAlgorithm(ModelFusionAlgorithm):
    def __init__(self, algorithm_config):
        super().__init__(algorithm_config)
        self.param = algorithm_config.get('param', 0.5)

algorithm = AlgorithmFactory.create_algorithm(config)

# New (v0.2+)
from fusion_bench.method import BaseAlgorithm

class NewAlgorithm(BaseAlgorithm):
    def __init__(self, param: float = 0.5, **kwargs):
        self.param = param
        super().__init__(**kwargs)

algorithm = instantiate(config)  # or direct instantiation
```


For backward compatibility, v0.1.x style configurations and factory methods are still supported through the `fusion_bench.compat` module, but new implementations should use the v0.2+ style.

## Implementation Details

- [fusion_bench.method.BaseAlgorithm][]
