# .vscode Folder

This folder contains configuration files for Visual Studio Code to enhance the development experience for the `fusion_bench` project.

## Files

### settings.json.template

This file includes settings for Python testing, search exclusions, and file exclusions.

- **Python Testing**: Configures `unittest` as the testing framework and specifies the test discovery pattern.
- **Search Exclusions**: Excludes certain directories and files from search results.
- **File Exclusions**: Excludes certain directories and files from the file explorer.

### launch.json.template

This file includes configurations for debugging the `fusion_bench.scripts.cli` module, i.e, the `fusion_bench` command-line interface (CLI).

- **Debug Configuration**: Sets up the `debugpy` debugger to launch the `fusion_bench.scripts.cli` module with specific arguments and environment variables.

## Usage

1. **Copy Templates**: Copy `settings.json.template` and `launch.json.template` to `settings.json` and `launch.json` respectively.
   
    ```shell
    cd .vscode

    cp settings.json.template settings.json
    cp launch.json.template launch.json
    ```

2. **Customize**: Modify the copied files as needed to fit your development environment.

    For example, you may want to add new debugging configurations for custom experiments.

    ```json
    {
        "configurations": [
            {
                "name": "Custom Experiment",
                "type": "debugpy",
                "request": "launch",
                "module": "fusion_bench.scripts.cli",
                "args": [
                    "--config-name custom_experiment",
                    "method=method_name",
                    "method.option_1=value_1",
                ],
                "env": {
                    "HYDRA_FULL_ERROR": "1",
                    "CUSTOM_ENV_VAR": "value"
                }
            }
        ]
    }
    ```

3. **Open in VS Code**: Open the project in Visual Studio Code to utilize the configurations.
