# fusion_bench.mixins

The mixins module provides reusable functionality through mixin classes that can be combined with other classes to add specific capabilities. These mixins follow the composition-over-inheritance principle and are designed to be modular, flexible, and easy to integrate.

## Basic Mixin Composition

```python
from fusion_bench.mixins import (
    LightningFabricMixin, 
    SimpleProfilerMixin,
    auto_register_config
)

@auto_register_config
class MyAlgorithm(
    LightningFabricMixin,
    SimpleProfilerMixin,
    BaseAlgorithm
):
    def __init__(self, learning_rate: float = 0.001, batch_size: int = 32, **kwargs):
        super().__init__(**kwargs)

    def run(self, modelpool):
        # ... implement the algorithm here
```

## Class Definitions

### Configuration and Instantiation
- [fusion_bench.mixins.HydraConfigMixin][]: A mixin class that provides configuration-based instantiation capabilities.
- [fusion_bench.mixins.auto_register_config][]: Decorator for automatically mapping constructor parameters to configuration keys.

### Serialization and Persistence  
- [fusion_bench.mixins.YAMLSerializationMixin][]: Provides methods for serializing and deserializing objects to and from YAML format.
- [fusion_bench.mixins.BaseYAMLSerializable][]: Base class for objects that support YAML serialization.

### Distributed Computing and Training
- [fusion_bench.mixins.LightningFabricMixin][]: Integrates with Lightning Fabric for automatic distributed environment and accelerator management.
- [fusion_bench.mixins.FabricTrainingMixin][]: Extends Lightning Fabric integration with training-specific utilities.

### Performance and Debugging
- [fusion_bench.mixins.SimpleProfilerMixin][]: Provides simple profiling capabilities for measuring execution time.
- [fusion_bench.mixins.PyinstrumentProfilerMixin][]: Offers advanced statistical profiling using the pyinstrument library.

### Computer Vision
- [fusion_bench.mixins.CLIPClassificationMixin][]: Supports CLIP-based image classification tasks.

## Class Decorators

- [fusion_bench.mixins.auto_register_config][]

## References

::: fusion_bench.mixins.HydraConfigMixin
::: fusion_bench.mixins.YAMLSerializationMixin
::: fusion_bench.mixins.BaseYAMLSerializable
::: fusion_bench.mixins.LightningFabricMixin
::: fusion_bench.mixins.FabricTrainingMixin
::: fusion_bench.mixins.SimpleProfilerMixin
::: fusion_bench.mixins.PyinstrumentProfilerMixin
::: fusion_bench.mixins.CLIPClassificationMixin
::: fusion_bench.mixins.auto_register_config
