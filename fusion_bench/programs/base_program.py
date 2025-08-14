"""
Base Program Classes for FusionBench.

This module defines the foundational abstract base classes for FusionBench programs.
These programs serve as the main execution units that orchestrate model fusion
workflows, from loading configurations to executing fusion algorithms and
evaluating results.

The base classes provide a consistent interface for all FusionBench programs
while allowing for flexible implementations of different fusion workflows.
"""

from abc import abstractmethod

from fusion_bench.mixins import BaseYAMLSerializable


class BaseHydraProgram(BaseYAMLSerializable):
    """
    Abstract base class for all FusionBench programs that use Hydra configuration.

    This class serves as the foundation for all FusionBench execution programs,
    providing a standardized interface for configuration-driven model fusion
    workflows. It combines the serialization capabilities of BaseYAMLSerializable
    with the requirement for a main execution method.

    The class is designed to work seamlessly with Hydra's configuration management
    system, allowing programs to be instantiated and configured through YAML files.
    This enables flexible, reproducible experiments with different fusion algorithms,
    model pools, and evaluation tasks.

    Key Features:

    - Configuration-driven execution through Hydra integration
    - YAML serialization support for experiment reproducibility
    - Abstract interface ensuring consistent program structure
    - Integration with FusionBench's modular architecture

    Typical Usage:
        Subclasses should implement the `run()` method to define their specific
        fusion workflow. The program can then be executed through the FusionBench
        CLI or instantiated directly from configuration files.

    Example:
        ```python
        class MyFusionProgram(BaseHydraProgram):
            def __init__(self, method_config, modelpool_config, taskpool_config):
                self.method_config = method_config
                self.modelpool_config = modelpool_config
                self.taskpool_config = taskpool_config

            def run(self):
                # Load components
                algorithm = load_algorithm(self.method_config)
                modelpool = load_modelpool(self.modelpool_config)
                taskpool = load_taskpool(self.taskpool_config)

                # Execute fusion
                merged_model = algorithm.run(modelpool)

                # Evaluate results
                report = taskpool.evaluate(merged_model)
                return report
        ```

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement the `run()` method to provide concrete
        functionality.

    See Also:

    - [FabricModelFusionProgram][fusion_bench.programs.FabricModelFusionProgram]: Lightning Fabric-based implementation
    - [BaseYAMLSerializable][fusion_bench.mixins.BaseYAMLSerializable]: Parent class providing serialization
    - FusionBench CLI documentation for program execution details
    """

    @abstractmethod
    def run(self):
        """
        Execute the main program workflow.

        This abstract method defines the primary entry point for program execution.
        Subclasses must implement this method to define their specific fusion
        workflow, including model loading, fusion algorithm execution, and
        result evaluation.
        """
        pass
