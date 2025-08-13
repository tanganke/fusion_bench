# CLIP Simple Average

This tutorial demonstrates how to merge the vision encoders of CLIP (Contrastive Language-Image Pre-training) models using the Simple Average algorithm - a straightforward, *hyperparameter-free* approach to model fusion that combines multiple task-specific models into a single unified model.

Mathematically, the Simple Average algorithm can be expressed as:

\[
\theta_{merged} = \frac{1}{N} \sum_{i=1}^{N} \theta_{i}
\]

where \( \theta_{merged} \) is the set of parameters for the merged model, \( N \) is the number of source models, and \( \theta_{i} \) are the parameters of the individual models.

This method works especially well for *large* models that have been fine-tuned on distinct downstream tasks[^3], or for models trained on the same task but with varying hyperparameter settings such as learning rate or batch size[^1][^2].

## 🔧 Standalone YAML Configuration

The example uses the following standalone configuration file that demonstrates merging CLIP models fine-tuned on different image classification datasets:

```yaml title="config/_get_started/clip_simple_average.yaml" linenums="1" hl_lines="1"
--8<-- "config/_get_started/clip_simple_average.yaml"
```

1. This is the program to handle the model fusion workflow.
2. This is the method config to perform model fusion.
3. This is the model pool config containing the base and fine-tuned models.
4. This is the task pool config defining evaluation datasets.

### Configuration Breakdown

- **Program**: This is the top level configuration that specifies the program to run. 
    Here we specify the main program as `FabricModelFusionProgram` which handles the model fusion workflow.
- **Method**: This is the method config to perform model fusion. 
    Here we specify the method as `SimpleAverageAlgorithm`, which performs model fusion.
- **Model Pool**: This is the model pool config containing the base and fine-tuned models. 
    In this example, it contains two fine-tuned models:
      - `sun397`: Fine-tuned on SUN397 scene recognition dataset
      - `stanford-cars`: Fine-tuned on Stanford Cars dataset
- **Task Pool**: A task pool object is responsible for evaluating the merged model's performance. In this example, we specify t

## 🚀 Running the Example

Execute the model merging process with the following command:

```bash
fusion_bench --config-path $PWD/config/_get_started --config-name clip_simple_average
```

This command will:

1. **Load the specified CLIP models** from the model pool
2. **Apply the Simple Average algorithm** to merge their parameters
3. **Evaluate the merged model** on the specified test datasets
4. **Generate performance reports** comparing the merged model against individual models

## 🎓 Key Learning Points

This example teaches you:

1. **Basic Configuration**: How to structure a FusionBench configuration file
2. **Model Pool Setup**: How to specify multiple models for merging
3. **Task Pool Configuration**: How to define evaluation datasets
4. **Simple Execution**: How to run model merging with a single command

## 🐛 Debugging Configuration (VS Code)

```json title=".vscode/launch.json"
{
    "name": "clip_simple_average",
    "type": "debugpy",
    "request": "launch",
    "module": "fusion_bench.scripts.cli",
    "args": [
        "--config-path",
        "${workspaceFolder}/config/_get_started",
        "--config-name",
        "clip_simple_average"
    ],
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
        "HYDRA_FULL_ERROR": "1"
    }
}
```

[^1]: M. Wortsman et al., “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” July 01, 2022, arXiv: arXiv:2203.05482. Accessed: May 08, 2023. Available: <http://arxiv.org/abs/2203.05482>
[^2]: A. Chegini et al., “Model Soup for Better RLHF: Weight Space Averaging to Improve Alignment in LLMs”.
[^3]: P. Yadav et al., “What Matters for Model Merging at Scale?,” Oct. 04, 2024, arXiv: arXiv:2410.03617. Accessed: Oct. 11, 2024. Available: <http://arxiv.org/abs/2410.03617>
