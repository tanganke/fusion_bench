# Merge Large Language Models using SLERP

This tutorial demonstrates how to merge Large Language Models (LLMs) using the [SLERP (Spherical Linear Interpolation)](../../algorithms/slerp.md) [^1] algorithm - a sophisticated model fusion technique that interpolates between two models along the surface of a high-dimensional sphere, preserving the geometric properties of the parameter space.

SLERP is particularly effective for merging language models because it maintains the angular relationships between model parameters, which can be crucial for preserving semantic representations and learned behaviors. Unlike simple linear interpolation (LERP), SLERP follows a curved path on the sphere's surface, ensuring consistent interpolation speed and avoiding potential distortions.

## 🔧 Standalone YAML Configuration

This example uses the following configuration that demonstrates merging LLMs using SLERP:

```yaml title="config/_get_started/llm_slerp.yaml" linenums="1" hl_lines="4-5"
--8<-- "config/_get_started/llm_slerp.yaml"
```

1. **Program Configuration**: Specifies [`FabricModelFusionProgram`][fusion_bench.programs.FabricModelFusionProgram] to handle the fusion workflow
2. **Method Configuration**: Uses [`SlerpForCausalLM`][fusion_bench.method.SlerpForCausalLM] with a scaling factor (t parameter), whose default value is set as 0.5. The option names in the configuration file are the same as those in the code.

    !!! note "[`SlerpForCausalLM.__init__()`][fusion_bench.method.SlerpForCausalLM.__init__]"

        ::: fusion_bench.method.SlerpForCausalLM.__init__
            options:
                show_root_heading: false
                parameter_headings: false
                heading_level: 4

3. **Model Pool**: Contains exactly two LLMs to be merged using spherical interpolation

## 🚀 Running the Example

Execute the SLERP fusion with the following command:

```bash
fusion_bench --config-path $PWD/config/_get_started --config-name llm_slerp
```

### Hyperparameter Tuning

You can experiment with different interpolation factors by overriding the configuration:

```bash
# Favor the first model more (closer to model_1)
fusion_bench --config-path $PWD/config/_get_started --config-name llm_slerp \
    method.t=0.3

# Balanced interpolation (default)
fusion_bench --config-path $PWD/config/_get_started --config-name llm_slerp \
    method.t=0.5

# Favor the second model more (closer to model_2)
fusion_bench --config-path $PWD/config/_get_started --config-name llm_slerp \
    method.t=0.7
```

## 🐛 Debugging Configuration (VS Code)

```json title=".vscode/launch.json"
{
    "name": "llm_slerp",
    "type": "debugpy",
    "request": "launch",
    "module": "fusion_bench.scripts.cli",
    "args": [
        "--config-path",
        "${workspaceFolder}/config/_get_started",
        "--config-name",
        "llm_slerp"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
        "HYDRA_FULL_ERROR": "1"
    }
}
```

[^1]: SLERP For Model Merging – A Primer https://www.coinfeeds.ai/ai-blog/slerp-model-merging-primer
