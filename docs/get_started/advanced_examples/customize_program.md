# Customize Program

This tutorial demonstrates how to create custom programs in FusionBench. While FusionBench is primarily designed for model fusion, programs are flexible execution units that can orchestrate **any type of workflow** - from simple greeting messages to complex data processing tasks, and of course, sophisticated model fusion experiments.

## 🎯 Understanding Programs

Programs in FusionBench serve as the **orchestration layer** that can:

- **Execute Any Workflow**: Not limited to model fusion - can be data processing, analysis, automation, etc.
- **Handle Configuration**: Parse and apply Hydra configurations for reproducible execution
- **Manage Resources**: Control logging, file I/O, and system resources
- **Coordinate Components**: When needed, manage interaction between algorithms, model pools, and task pools
- **Process Results**: Save outputs, generate reports, and handle workflow results

Programs provide the infrastructure to run **any configurable workflow** using FusionBench's robust configuration system.

The main program class is `FabricModelFusionProgram`, which integrates with PyTorch Lightning Fabric for scalable execution.

## 🏗️ Program Architecture

### Base Classes

All programs inherit from `BaseHydraProgram`:

```python
from fusion_bench.programs import BaseHydraProgram

class BaseHydraProgram(BaseYAMLSerializable):
    """
    Abstract base class for all FusionBench programs that use Hydra configuration.
    """
    
    @abstractmethod
    def run(self):
        """Execute the main program workflow."""
        pass
```

### FabricModelFusionProgram Structure

The main program class provides comprehensive functionality:

```python
from fusion_bench.programs import FabricModelFusionProgram
from fusion_bench.mixins import LightningFabricMixin

class FabricModelFusionProgram(
    LightningFabricMixin,  # Provides Lightning Fabric integration
    BaseHydraProgram,      # Provides Hydra configuration support
):
    # Core components
    method: BaseAlgorithm      # The fusion algorithm
    modelpool: BaseModelPool   # Collection of models to merge
    taskpool: BaseTaskPool     # Evaluation tasks (optional)
```

## 🔧 Creating Custom Programs

Custom programs don't have to be fusion programs! They can be any workflow that benefits from Hydra configuration management. Let's start with a simple example.

### Simple Greeting Program

Here's a minimal example that just prints a greeting message:

```python
import logging
from typing import Optional

from omegaconf import DictConfig

from fusion_bench.programs import BaseHydraProgram

log = logging.getLogger(__name__)


class GreetingProgram(BaseHydraProgram):
    """
    A simple program that greets users with a custom message.
    """

    _config_mapping = BaseHydraProgram._config_mapping | {
        "message": "message",
        "name": "name",
        "repeat_count": "repeat_count",
    }

    def __init__(
        self,
        message: str = "Hello",
        name: str = "World",
        repeat_count: int = 1,
        **kwargs,
    ):
        self.message = message
        self.name = name
        self.repeat_count = repeat_count
        super().__init__(**kwargs)

    def run(self):
        """Execute the greeting workflow."""
        log.info("Starting greeting program")

        # Create the greeting
        greeting = f"{self.message}, {self.name}!"

        # Print the greeting multiple times
        for i in range(self.repeat_count):
            if self.repeat_count > 1:
                print(f"[{i+1}/{self.repeat_count}] {greeting}")
            else:
                print(greeting)

        log.info("Greeting program completed")
        return greeting
```

**Usage Configuration:**

```yaml title="config/_get_started/greeting_program.yaml"
--8<-- "config/_get_started/greeting_program.yaml"
```

**Command Line Usage:**

```bash
fusion_bench \
    --config-path $PWD/config/_get_started \
    --config-name greeting_program \
    message="Hello there" \
    name="FusionBench User"
```

This program will output:

```text
[INFO] - Starting greeting program
[1/3] Hello there, FusionBench User!
[2/3] Hello there, FusionBench User!
[3/3] Hello there, FusionBench User!
[INFO] - Greeting program completed
```

### Basic Custom Program Template

Here's a template for creating your own program:

```python
import logging
from typing import Optional, Dict, Any
from omegaconf import DictConfig
from torch import nn

from fusion_bench.programs import BaseHydraProgram
from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.taskpool import BaseTaskPool
from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)


class CustomFusionProgram(BaseHydraProgram):
    """
    Custom program for specialized fusion workflows.
    """
    
    _config_mapping = BaseHydraProgram._config_mapping | {
        "_method": "method",
        "_modelpool": "modelpool", 
        "_taskpool": "taskpool",
        "custom_param": "custom_param",
    }
    
    def __init__(
        self,
        method: DictConfig,
        modelpool: DictConfig,
        taskpool: Optional[DictConfig] = None,
        custom_param: str = "default_value",
        **kwargs
    ):
        self._method = method
        self._modelpool = modelpool
        self._taskpool = taskpool
        self.custom_param = custom_param
        super().__init__(**kwargs)
    
    def run(self):
        """Execute the custom fusion workflow."""
        log.info("Starting custom fusion program")
        
        # 1. Load components
        self.method = instantiate(self._method)
        self.modelpool = instantiate(self._modelpool)
        
        if self._taskpool is not None:
            self.taskpool = instantiate(self._taskpool)
        
        # 2. Custom pre-processing
        self._preprocess_models()
        
        # 3. Execute fusion
        merged_model = self.method.run(self.modelpool)
        
        # 4. Custom post-processing
        merged_model = self._postprocess_model(merged_model)
        
        # 5. Evaluate if taskpool is available
        if hasattr(self, 'taskpool') and self.taskpool is not None:
            report = self.taskpool.evaluate(merged_model)
            self._save_report(report)
        
        return merged_model
    
    def _preprocess_models(self):
        """Custom preprocessing of models before fusion."""
        log.info("Preprocessing models...")
        # Add your custom preprocessing logic here
        pass
    
    def _postprocess_model(self, merged_model: nn.Module) -> nn.Module:
        """Custom postprocessing of the merged model."""
        log.info("Postprocessing merged model...")
        # Add your custom postprocessing logic here
        return merged_model
    
    def _save_report(self, report: Dict[str, Any]):
        """Save evaluation report with custom formatting."""
        log.info("Saving evaluation report...")
        # Add custom report saving logic here
        pass
```

## 🚀 Usage Examples

### Command Line Usage

```bash
# Use custom program
fusion_bench \
    --config-path my_configs \
    --config-name custom_fusion \
    program.custom_param="new_value"
```

### Programmatic Usage

```python
from omegaconf import DictConfig
from my_package import CustomFusionProgram

# Create configuration
config = DictConfig({
    "_target_": "mypackage.CustomProgram"
    "method": {...},
    "modelpool": {...},
})

# Instantiate and run program
program = CustomFusionProgram(**config)
result = program.run()
```
