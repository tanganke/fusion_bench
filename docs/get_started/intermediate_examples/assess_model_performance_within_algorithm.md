# Assess Model Performance During Algorithm Execution

This tutorial demonstrates how to evaluate model performance during the merging process using FusionBench's TaskPool system. You'll learn how to integrate evaluation at different stages of your algorithm, monitor performance throughout the merging process, and save intermediate results for analysis.

## 🎯 Overview

During model merging, it's often valuable to assess how performance changes as models are incrementally merged. This can help you:

- **Monitor Progress**: Track how performance evolves during merging
- **Early Stopping**: Stop merging if performance starts degrading
- **Compare Strategies**: Evaluate different merging orders or parameters
- **Debug Issues**: Identify when and why merging performance drops
- **Research Insights**: Understand the dynamics of model fusion

## 🏗️ TaskPool Integration

### Understanding TaskPools

TaskPools in FusionBench manage evaluation datasets and provide standardized interfaces for assessing model performance. The most common is `CLIPVisionModelTaskPool` for vision models:

```python
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from copy import deepcopy

# Access taskpool from the program context
taskpool = self._program.taskpool  # Available in algorithm classes
```

### Basic Evaluation Pattern

Here's the fundamental pattern for evaluating during algorithm execution:

```python linenums="1" hl_lines="17 39-40 71"
import torch
from copy import deepcopy
from pathlib import Path
from fusion_bench import BaseAlgorithm
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.json import save_to_json

class EvaluatingMergingAlgorithm(BaseAlgorithm):
    
    def __init__(self, evaluate_on_every_step: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_on_every_step = evaluate_on_every_step
    
    @torch.no_grad()
    def run(self, modelpool):
        # Access the taskpool from the program
        taskpool = self._program.taskpool
        
        # Store original test datasets for restoration
        original_test_datasets = deepcopy(taskpool._test_datasets)
        
        model_names = modelpool.model_names
        merged_model = modelpool.load_model(model_names[0])
        
        # Evaluate initial model
        if self.evaluate_on_every_step:
            report = self._evaluate_model(taskpool, merged_model, model_names[0], step=0)
            
        # Iterative merging with evaluation
        for step, model_name in enumerate(model_names[1:], 1):
            # Load and merge next model
            next_model = modelpool.load_model(model_name)
            merged_model = self._merge_models(merged_model, next_model)
            
            # Evaluate merged model
            if self.evaluate_on_every_step:
                # Update taskpool to include models merged so far
                current_models = model_names[:step + 1]
                report = self._evaluate_model(
                    taskpool, merged_model, current_models, step
                )
        
        # Restore original taskpool state
        taskpool._test_datasets = original_test_datasets
        taskpool._is_setup = False
        
        return merged_model
    
    def _evaluate_model(self, taskpool, model, model_names, step):
        """Evaluate model and save results."""
        # Reset taskpool setup to reconfigure with new datasets
        taskpool._is_setup = False
        
        # Configure taskpool for current set of models
        if isinstance(model_names, list):
            # Multiple models - evaluate on their respective datasets
            current_datasets = {
                name: taskpool._test_datasets[name] 
                for name in model_names 
                if name in taskpool._test_datasets
            }
        else:
            # Single model
            current_datasets = {model_names: taskpool._test_datasets[model_names]}
        
        # Update taskpool configuration
        from omegaconf import DictConfig
        taskpool._test_datasets = DictConfig(current_datasets)
        
        # Run evaluation
        report = taskpool.evaluate(deepcopy(model))
        
        # Save results
        if hasattr(self, 'log_dir') and self.log_dir:
            save_path = Path(self.log_dir) / f"report_{step}.json"
            save_to_json(report, save_path)
        
        return report
    
    def _merge_models(self, model1, model2):
        """Implement your merging logic here."""
        # This is a placeholder - implement your actual merging algorithm
        pass
```

## 🔍 Real-World Example: OPCM Algorithm

The Orthogonal Projection-based Continual Merging (OPCM) algorithm provides an excellent example of evaluation during merging. For more information, refer to the [OPCM implementation][fusion_bench.method.OPCMForCLIP].
