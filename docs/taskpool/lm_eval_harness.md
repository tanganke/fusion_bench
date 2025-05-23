# Language Model Evaluation Harness Task Pool

The `LMEvalHarnessTaskPool` is a task pool implementation that integrates the [LM-Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness) library into the Fusion Bench framework. It allows you to evaluate language models on a wide range of standardized benchmarks and tasks.

## Usage

### Basic Usage

```bash
fusion_bench \
    method=dummy \
    modelpool=CausalLMPool/single_llama_model \
    taskpool=LMEvalHarnessTaskPool/lm_eval \
    taskpool.tasks="hellaswag,truthfulqa"
```

### Configuration Options

The `LMEvalHarnessTaskPool` supports the following configuration options:

| Parameter             | Type                                                               | Default  | Description                                                                                                                                                                 |
| --------------------- | ------------------------------------------------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tasks`               | Union[str, List[str]]                                              | Required | Comma-separated list of task names or a list of task names                                                                                                                  |
| `apply_chat_template` | bool                                                               | False    | Whether to apply chat template to the prompts                                                                                                                               |
| `include_path`        | Optional[str]                                                      | None     | Additional path to include for external tasks                                                                                                                               |
| `batch_size`          | int                                                                | 1        | Batch size for model evaluation                                                                                                                                             |
| `metadata`            | Optional[DictConfig]                                               | None     | Additional metadata to pass to task configs                                                                                                                                 |
| `verbosity`           | Optional[Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]] | None     | Logging verbosity level                                                                                                                                                     |
| `output_path`         | Optional[str]                                                      | None     | Path to save evaluation results, if not specified, the results will be saved to `log_dir/lm_eval_results`, where `log_dir` is the directory controlled by lightning Fabric. |
| `log_samples`         | bool                                                               | False    | Whether to log individual samples                                                                                                                                           |

### Example Configurations

Basic evaluation with multiple tasks:

```bash
fusion_bench \
    method=dummy \
    modelpool=CausalLMPool/single_llama_model \
    taskpool=LMEvalHarnessTaskPool/lm_eval \
    taskpool.tasks="hellaswag,truthfulqa"
```

Here `dummy` method simply loads the pre-trained model or the first model in the model pool and does nothing else.

## Available Tasks

To see a complete list of available tasks, you can use:
```bash
lm-eval --tasks list
```
