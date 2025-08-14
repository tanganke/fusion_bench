# Evaluate a Single CLIP Model

This tutorial demonstrates how to evaluate a single CLIP (Contrastive Language-Image Pre-training) model on multiple downstream vision tasks using FusionBench CLI. 
This serves as a baseline for understanding model performance before applying fusion techniques.

This example utilizes the [`DummyAlgorithm`][fusion_bench.method.DummyAlgorithm], a specialized class designed for single model evaluation. It returns the pretrained model as-is, or the first available model if `_pretrained_` is not present, without applying any modifications.

## 🔧 Standalone YAML Configuration

The example uses the following configuration that evaluates a pretrained CLIP model on multiple image classification datasets:

```yaml title="config/_get_started/clip_evaluate_single_model.yaml" linenums="1" hl_lines="8"
--8<-- "config/_get_started/clip_evaluate_single_model.yaml"
```

1. **Program Configuration**: Specifies [`FabricModelFusionProgram`][fusion_bench.programs.FabricModelFusionProgram] to handle the evaluation workflow
2. **Method Configuration**: Uses [`DummyAlgorithm`][fusion_bench.method.DummyAlgorithm] which passes through the input model unchanged
3. **Model Pool**: Contains only the base pretrained CLIP model (`openai/clip-vit-base-patch32`).
    ```python
    models={'_pretrained_': 'openai/clip-vit-base-patch32'}
    ```
4. **Task Pool**: Defines evaluation datasets and specifies the CLIP model and processor for inference.
    In this examples:
    ```python
    test_datasets = {
        'sun397': ...,
        'stanford-cars': ...,
    }
    ```

## 🚀 Running the Example

Execute the model evaluation with the following command:

```bash
fusion_bench \
    --config-path $PWD/config/_get_started \
    --config-name clip_evaluate_single_model
```

Or override the model path via pass `modelpool.models._pretrained_=<new_model_path>`:

```bash
fusion_bench \
    --config-path $PWD/config/_get_started \
    --config-name clip_evaluate_single_model \
    modelpool.models._pretrained_=<new_model_path>
```

## 🐛 Debugging Configuration (VS Code)

```json title=".vscode/launch.json"
{
    "name": "clip_evaluate_single_model",
    "type": "debugpy",
    "request": "launch",
    "module": "fusion_bench.scripts.cli",
    "args": [
        "--config-path",
        "${workspaceFolder}/config/_get_started",
        "--config-name",
        "clip_evaluate_single_model"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
        "HYDRA_FULL_ERROR": "1"
    }
}
```
