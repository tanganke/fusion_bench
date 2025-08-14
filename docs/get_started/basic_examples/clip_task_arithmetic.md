# CLIP Task Arithmetic

This tutorial demonstrates how to merge CLIP (Contrastive Language-Image Pre-training) models using the [Task Arithmetic algorithm](../../algorithms/task_arithmetic.md) [^1] - a powerful model fusion technique that combines multiple task-specific models by manipulating their "task vectors" with configurable scaling factors.

Task Arithmetic is an advanced model fusion technique that operates on the concept of **task vectors** - the directional differences between a fine-tuned model and its pretrained base model. This approach provides more fine-grained control over the fusion process compared to simple averaging.

Mathematically, Task Arithmetic can be expressed as:

**Step 1: Compute Task Vectors**

\[
\tau_i = \theta_i - \theta_0
\]

**Step 2: Scale and Combine Task Vectors**

\[
\theta_{merged} = \theta_0 + \lambda \sum_{i=1}^{N} \tau_i
\]

where:

- \( \theta_{merged} \) is the final merged model parameters
- \( \theta_0 \) is the pretrained base model parameters  
- \( \theta_i \) are the fine-tuned model parameters
- \( \tau_i \) are the task vectors (learned adaptations)
- \( \lambda \) is the scaling factor that controls the strength of task vector influence
- \( N \) is the number of task-specific models

## 🔧 Standalone YAML Configuration

The example uses the following configuration that demonstrates merging CLIP models with task arithmetic on image classification datasets:

```yaml title="config/_get_started/clip_task_arithmetic.yaml" linenums="1" hl_lines="5"
--8<-- "config/_get_started/clip_task_arithmetic.yaml"
```

1. **Program Configuration**: Specifies [`FabricModelFusionProgram`][fusion_bench.programs.FabricModelFusionProgram] to handle the fusion workflow
2. **Method Configuration**: Uses [`TaskArithmeticAlgorithm`][fusion_bench.method.TaskArithmeticAlgorithm] with a scaling factor, whose default value is set as 0.7. The option names in the configuration file are the same as those in the code.

    !!! note "[`TaskArithmeticAlgorithm.__init__()`][fusion_bench.method.TaskArithmeticAlgorithm.__init__]"

        ::: fusion_bench.method.TaskArithmeticAlgorithm.__init__
            options:
                show_root_heading: false
                parameter_headings: false
                heading_level: 4

3. **Model Pool**: Contains the base pretrained model and fine-tuned variants
4. **Task Pool**: Defines evaluation datasets for performance assessment

## 🚀 Running the Example

Execute the task arithmetic fusion with the following command:

```bash
fusion_bench --config-path $PWD/config/_get_started --config-name clip_task_arithmetic
```

### Hyperparameter Tuning

You can experiment with different scaling factors by overriding the configuration:

```bash
# More conservative fusion (less task-specific influence)
fusion_bench --config-path $PWD/config/_get_started --config-name clip_task_arithmetic \
    method.scale_factor=0.5

# More aggressive fusion (stronger task-specific influence)  
fusion_bench --config-path $PWD/config/_get_started --config-name clip_task_arithmetic \
    method.scale_factor=1.0
```

## 🐛 Debugging Configuration (VS Code)

```json title=".vscode/launch.json"
{
    "name": "clip_task_arithmetic",
    "type": "debugpy",
    "request": "launch",
    "module": "fusion_bench.scripts.cli",
    "args": [
        "--config-path",
        "${workspaceFolder}/config/_get_started",
        "--config-name",
        "clip_task_arithmetic"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
        "HYDRA_FULL_ERROR": "1"
    }
}
```

[^1]: G. Ilharco et al., “Editing Models with Task Arithmetic,” Mar. 31, 2023, arXiv: arXiv:2212.04089. doi: 10.48550/arXiv.2212.04089.
