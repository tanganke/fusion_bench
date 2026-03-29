# fusion_bench.program

The `fusion_bench.programs` module provides execution frameworks for running fusion benchmarks. These programs orchestrate the entire fusion pipeline from algorithm execution to evaluation.

## Class Definitions

### Base Program

[`BaseHydraProgram`][fusion_bench.programs.BaseHydraProgram] - Base class for Hydra-based execution programs.

### Model Fusion Program

[`FabricModelFusionProgram`][fusion_bench.programs.FabricModelFusionProgram] - Program for fusing models using Lightning Fabric.

## Usage Example

```python
from fusion_bench import (
    BaseAlgorithm,
    BaseModelPool,
    FabricModelFusionProgram
)

# Define algorithm and model pool
algorithm = BaseAlgorithm.from_yaml("config/method/simple_average.yaml")
modelpool = BaseModelPool.from_yaml("config/modelpool/clip_vit.yaml")

# Create and run program
program = FabricModelFusionProgram(algorithm=algorithm, modelpool=modelpool)
program.run()
```

## API Reference

::: fusion_bench.programs.BaseHydraProgram
    options:
        show_root_heading: true

::: fusion_bench.programs.FabricModelFusionProgram
    options:
        show_root_heading: true

::: fusion_bench.programs.ModelFusionProgram
    options:
        show_root_heading: true

---

## Related Documentation

- [Get Started: Basic Examples](../get_started/basic_examples/index.md) - Simple usage examples
- [Get Started: Intermediate Examples](../get_started/intermediate_examples/index.md) - Advanced configurations
- [CLI Reference](../cli/fusion_bench.md) - Command-line interface documentation
