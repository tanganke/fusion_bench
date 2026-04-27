# `fusion_bench`: The Command Line Interface for FusionBench

`fusion_bench` is the command line interface for running model fusion benchmarks in the FusionBench project.
It provides a flexible way to configure and execute various fusion algorithms on different model pools and evaluate them across multiple tasks.

## Details and Options

`fusion_bench` takes a configuration file as input, which specifies the models, fusion method to be used, and the datasets to be evaluated. Running `fusion_bench` is equivalent to running `python fusion_bench/scripts/cli.py`.

```bash
fusion_bench [--config-path CONFIG_PATH] [--config-name CONFIG_NAME] \
    OPTION_1=VALUE_1 OPTION_2=VALUE_2 ...

# or equivalently
python fusion_bench/scripts/cli.py [--config-path CONFIG_PATH] [--config-name CONFIG_NAME] \
    OPTION_1=VALUE_1 OPTION_2=VALUE_2 ...
```

The entry point is defined in `fusion_bench/scripts/cli.py` using Hydra's `@hydra.main` decorator. The default config path is `config/` and the default config name is `fabric_model_fusion.yaml`.

```python
@hydra.main(
    config_path=get_default_config_path(),
    config_name="fabric_model_fusion",
    version_base=None,
)
def _hydra_main(cfg: DictConfig) -> None:
    main(cfg)
```

The `main()` function resolves config interpolations, instantiates the program via `fusion_bench.utils.instantiate`, and calls `program.run()`.

## Hydra Options

### Help and Info

- **--help, -h**: Application's help. Print help message and exit.

  ```bash
  fusion_bench --help
  ```

- **--hydra-help**: Hydra's help.
- **--version**: Show Hydra's version and exit.
- **--info, -i**: Print Hydra information. Accepts: `all`, `config`, `defaults`, `defaults-tree`, `plugins`, `searchpath`.

  ```bash
  fusion_bench --info defaults-tree
  ```

### Configuration Inspection

- **--cfg, -c**: Show config instead of running. Accepts: `job`, `hydra`, or `all` (default).
  This prints plain text configuration without color highlighting.

  ```bash
  fusion_bench --cfg job
  ```

  For a beautifully formatted output with syntax highlighting, use:

  ```bash
  fusion_bench print_config=true dry_run=true
  ```

- **--resolve**: Used in conjunction with `--cfg`, resolves config interpolations (e.g., `${path.data_dir}`) before printing.

  ```bash
  fusion_bench --cfg job --resolve
  ```

- **--package, -p**: Config package to show. For example, when you only want to see the configuration for `method`:

  ```bash
  fusion_bench --cfg job -p method
  ```

### Config Path Overrides

- **--config-path, -cp**: Overrides the `config_path` specified in `@hydra.main()`. The path is absolute or relative to the Python file declaring the decorator (i.e., `fusion_bench/scripts/cli.py`).
  By default, the config path is the `config` directory in the project root.

  ```bash
  fusion_bench --config-path /path/to/custom/configs method=my_method
  ```

- **--config-name, -cn**: Overrides the `config_name` specified in `@hydra.main()`. By default, the config name is `fabric_model_fusion` so `config/fabric_model_fusion.yaml` is loaded.

  ```bash
  fusion_bench --config-name custom_config.yaml
  ```

- **--config-dir, -cd**: Adds an additional config directory to the config search path.

  ```bash
  fusion_bench --config-dir ./my_configs method=my_method modelpool=my_models
  ```

### Multi-Run and Sweeps

- **--multirun, -m**: Run multiple jobs with the configured launcher and sweeper. This is the primary way to run hyperparameter sweeps or compare multiple configurations in a single command.

  See [Hydra multi-run documentation](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/) for details.

  ```bash
  # Sweep scaling_factor across values
  fusion_bench -m \
    method=task_arithmetic \
    method.scaling_factor=0.1,0.3,0.5,1.0 \
    modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
    taskpool=dummy

  # Sweep across methods
  fusion_bench -m \
    method=task_arithmetic,simple_average \
    modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
    taskpool=dummy
  ```

- **--experimental-rerun**: Rerun a job from a previous config pickle.

### Shell Completion

Install shell completion for tab-completing arguments:

- **Bash - Install**:

  ```bash
  eval "$(fusion_bench -sc install=bash)"
  ```

- **Bash - Uninstall**:

  ```bash
  eval "$(fusion_bench -sc uninstall=bash)"
  ```

- **Fish - Install**:

  ```fish
  fusion_bench -sc install=fish | source
  ```

- **Fish - Uninstall**:

  ```fish
  fusion_bench -sc uninstall=fish | source
  ```

- **Zsh - Install/Uninstall**: Compatible with Bash shell completion.

  ```zsh
  eval "$(fusion_bench -sc install=bash)"
  ```

## Application Options

These options are defined in the main configuration file (`config/fabric_model_fusion.yaml`) and can be overridden from the command line.

### Execution Control

- **dry_run**: Perform a dry run. Validates the configuration without running the actual fusion code. Default is `false`.

  ```bash
  fusion_bench dry_run=true print_config=true
  ```

- **fast_dev_run**: Quick testing mode. Runs on a single batch instead of the full dataset. Default is `false`. This is invaluable during development for rapid iteration.

  ```bash
  fusion_bench --fast_dev_run
  # or
  fusion_bench fast_dev_run=true
  ```

- **print_config**: Whether to print the resolved configuration before execution. Default is `true`.

  ```bash
  fusion_bench print_config=true
  ```

- **print_function_call**: Show detailed instantiation calls. Default is `true`.

  ```bash
  fusion_bench print_function_call=false
  ```

- **seed**: Random seed for reproducibility. Default is `null` (no seed set).

  ```bash
  fusion_bench seed=42
  ```

### Output and Logging

- **report_save_path**: Path to save the evaluation report as a JSON file. Default is `{log_dir}/program_report.json`. Set to `false` to skip saving.

  ```bash
  fusion_bench report_save_path=outputs/my_report.json
  # or disable
  fusion_bench report_save_path=false
  ```

- **merged_model_save_path**: Path to save the merged model. If specified, calls `modelpool.save_model(merged_model, path)`. Default is `null` (no save).

  ```bash
  fusion_bench merged_model_save_path=outputs/merged_model.pt
  ```

  The exact save behavior depends on the ModelPool implementation:
  - **BaseModelPool**: Saves `model.state_dict()` as a `.pt` file.
  - **CLIPVisionModelPool**: Calls `model.save_pretrained(path)` (saves a directory).
  - **ResNetForImageClassificationPool**: Saves state dict for torchvision models, `save_pretrained` for transformers models.
  - **CausalLMPool**: Calls `model.save_pretrained(path)` with optional tokenizer saving.

- **merged_model_save_kwargs**: Extra keyword arguments forwarded to `modelpool.save_model()`. Provide a dict-like value using YAML-style braces.

  ```bash
  fusion_bench \
    merged_model_save_path=outputs/merged_model \
    merged_model_save_kwargs='{push_to_hub: false, save_tokenizer: true}'
  ```

## Core Components: method, modelpool, taskpool

These three options define the fusion experiment:

```bash
fusion_bench method=<METHOD> modelpool=<MODELPOOL> taskpool=<TASKPOOL>
```

- **method**: Fusion algorithm (e.g., `simple_average`, `task_arithmetic`, `adamerging/clip`)
- **modelpool**: Model pool configuration defining which models to merge
- **taskpool**: Task pool configuration defining evaluation tasks

### Parameter Overrides

Use dot-notation to override nested configuration values:

```bash
fusion_bench \
  method=task_arithmetic \
  method.scaling_factor=0.7 \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val
```

For deeply nested overrides:

```bash
fusion_bench \
  method=task_arithmetic \
  method.optimizer.lr=0.001 \
  method.optimizer.weight_decay=0.01
```

To append to lists or set specific indices:

```bash
fusion_bench taskpool.test_datasets.0.my_field=new_value
```

## Hydra Override Syntax

Hydra provides powerful syntax for configuration overrides:

### Adding to Dictionaries

```bash
# Add a new key-value pair
fusion_bench +method.new_param=value

# Add a new nested section
fusion_bench ++modelpool.new_section='{key: value}'
```

### Removing Keys

```bash
# Remove a config key
fusion_bench ~method.unwanted_param
```

### Special Values

```bash
# Set to null
fusion_bench method.some_param=null

# Set boolean
fusion_bench dry_run=true

# Set float with scientific notation
fusion_bench method.learning_rate=1e-4
```

### Null vs False

```bash
# Setting to null (YAML null)
fusion_bench merged_model_save_path=null

# Setting to false (YAML boolean)
fusion_bench report_save_path=false
```

## Multi-Run with Hydra (Sweeps)

Hydra's multi-run mode (`-m`) enables parameter sweeps. Combine with the `launchers` configuration for parallel execution.

### Basic Sweep

```bash
fusion_bench -m method.scaling_factor=0.1,0.3,0.5,1.0
```

This runs 4 jobs, one for each value.

### Grid Sweep

```bash
fusion_bench -m \
  method.scaling_factor=0.1,0.5,1.0 \
  method.another_param=true,false
```

This runs 6 jobs (3 x 2 combinations).

### Using Hydra Launcher

For parallel execution with joblib:

```yaml
# In your Hydra config
hydra:
  launcher:
    _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
```

Or use the built-in parallel launcher:

```bash
fusion_bench -m hydra.launcher.n_jobs=4 method.scaling_factor=0.1,0.3,0.5,1.0
```

### Output Directory Structure

Multi-run jobs are organized in timestamped directories:

```
outputs/
  fusion_bench/
    2024-01-15/
      12-00-00/
        job0/    # scaling_factor=0.1
        job1/    # scaling_factor=0.3
        job2/    # scaling_factor=0.5
        job3/    # scaling_factor=1.0
```

Use `${now:...}` interpolation in paths for custom organization:

```bash
fusion_bench -m \
  report_save_path=outputs/sweep_\${now:%Y%m%d}/\${method_name}/report.json \
  method.scaling_factor=0.1,0.3,0.5
```

## Basic Examples

### Merge Two CLIP Models Using Task Arithmetic

```bash
fusion_bench method=task_arithmetic \
  modelpool=clip-vit-base-patch32_svhn_and_mnist \
  taskpool=clip-vit-base-patch32_svhn_and_mnist
```

### Merge Multiple CLIP Models Using Simple Averaging

```bash
fusion_bench method=simple_average \
  modelpool=clip-vit-base-patch32_TA8.yaml \
  taskpool=dummy
```

### Full Experiment with Model Saving

```bash
fusion_bench \
  method=task_arithmetic \
  method.scaling_factor=0.5 \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  merged_model_save_path=outputs/final_model \
  report_save_path=outputs/final_report.json \
  seed=42
```

## Running in Offline Mode

In offline mode, the model pool does not download models from the internet. Instead, it uses models from the local cache.

```bash
source offline_mode.sh
```

Or set the environment variable according to the content of `offline_mode.sh`.

## Debugging and Troubleshooting

### Quick Validation

Use `--cfg` to inspect the resolved configuration without running anything:

```bash
fusion_bench --cfg job method=task_arithmetic modelpool=my_pool taskpool=dummy
```

Use `--resolve` to expand interpolations:

```bash
fusion_bench --cfg job --resolve
```

### Fast Development Cycle

Use `--fast_dev_run` for rapid iteration during algorithm development:

```bash
fusion_bench --fast_dev_run method=task_arithmetic ...
```

This evaluates on a single batch, drastically reducing runtime while still testing the full pipeline.

### Debugging in VSCode

Configure `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FusionBench with Arguments",
            "type": "debugpy",
            "request": "launch",
            "module": "fusion_bench.scripts.cli",
            "args": "${command:pickArgs}",
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
```

The `module` field specifies `fusion_bench.scripts.cli`. Use `${command:pickArgs}` to pick arguments interactively, or hardcode them:

```json
"args": ["method=simple_average", "modelpool=my_pool", "taskpool=dummy"]
```

### Debugging in PyCharm

1. Click `Run` > `Edit Configurations...`
2. Click `+` and select `Python`
3. Set:
   - **Script path**: Absolute path to `fusion_bench/scripts/cli.py`
   - **Parameters**: `method=simple_average modelpool=my_pool taskpool=dummy`
   - **Python interpreter**: Your project's virtual environment

### Common Errors

- **Missing `_target_`**: Every instantiated component needs a `_target_` field pointing to its Python class. Check the error message for the missing component.
- **`KeyError` on model name**: The model name does not exist in the pool. Use `--cfg job` to inspect available models.
- **`ValidationError` on model name**: Model names must follow naming conventions. Special names (`_pretrained_`, `_merged_`) must start and end with underscores.
- **Import errors with lazy loading**: FusionBench uses `LazyImporter` to defer heavy imports. If you see import errors, ensure dependencies are installed: `pip install fusion-bench[all]`.

### Logging

FusionBench uses Python's `logging` module. Increase verbosity by setting the `LOG_LEVEL` environment variable:

```bash
LOG_LEVEL=DEBUG fusion_bench method=task_arithmetic ...
```
