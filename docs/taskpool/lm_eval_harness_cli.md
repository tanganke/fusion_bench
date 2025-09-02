# LM Evaluation Harness CLI Tool

## Overview

The `scripts/lm_eval/evaluate_task.sh` script is a comprehensive command-line tool for evaluating language models on various tasks using the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework. This script provides a convenient wrapper around the lm_eval library with additional features like GPU detection, automated result organization, and support for multiple inference backends.

## Installation Requirements

Before using the script, ensure you have the following dependencies installed:

```bash
# Install LM Evaluation Harness
pip install -e '.[lm-eval-harness]'
```

## Basic Usage

### Syntax

```bash
./scripts/lm_eval/evaluate_task.sh MODEL --tasks TASK --output_path OUTPUT_DIR [OPTIONS...]
```

### Required Arguments

- `MODEL`: The model path or name to evaluate (positional argument)
- `--tasks TASK`: The task(s) to evaluate on (single task or comma-separated list)
- `--output_path OUTPUT_DIR`: Directory to save evaluation results

### Optional Arguments

- `--batch_size BATCH_SIZE`: Batch size for evaluation (default: auto)
- `--use_vllm`: Enable vLLM for optimized inference (default: false)
- `--help` or `-h`: Display help information

## Examples

### Single Task Evaluation

Evaluate a model on a single task with standard inference:

```bash
./scripts/lm_eval/evaluate_task.sh 'meta-llama/Llama-2-7b-hf' \
    --tasks 'hellaswag' \
    --output_path './outputs/lm_eval' \
    --batch_size 8 \
    --num_fewshot 5
```

### Multiple Tasks Evaluation

Evaluate a model on multiple tasks simultaneously:

```bash
./scripts/lm_eval/evaluate_task.sh 'meta-llama/Llama-2-7b-hf' \
    --tasks 'gsm8k,gsm8k_cot,hellaswag' \
    --output_path './outputs/lm_eval' \
    --batch_size 8
```

### Using vLLM for Optimized Inference

For faster inference with large models, use vLLM:

```bash
./scripts/lm_eval/evaluate_task.sh 'meta-llama/Llama-2-7b-hf' \
    --tasks 'lambada_openai' \
    --output_path './outputs/lm_eval' \
    --use_vllm \
    --batch_size auto
```

### Custom vLLM Configuration

Override default vLLM parameters:

```bash
./scripts/lm_eval/evaluate_task.sh 'meta-llama/Llama-2-7b-hf' \
    --tasks 'lambada_openai' \
    --output_path './outputs/lm_eval' \
    --use_vllm \
    --model_args 'pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=2,gpu_memory_utilization=0.9'
```

## Configuration Options

### Inference Backends

#### Standard Inference (Default)

When using standard inference, the script automatically configures:

- Model type: `hf` (Hugging Face transformers)
- Default model arguments: `pretrained=$MODEL,dtype=bfloat16,parallelize=True`

#### vLLM Inference

When `--use_vllm` is enabled, the script configures:

- Model type: `vllm`
- Default parameters:
  - `tensor_parallel_size=1`
  - `dtype=auto`
  - `gpu_memory_utilization=0.8`
  - `data_parallel_size=1`

## Output Structure

Results are organized in the following directory structure:

```
OUTPUT_DIR/
├── TASK_NAME_1/
│   └── MODEL_NAME__SANITIZED/
│       ├── results.json
│       └── samples/
└── TASK_NAME_2/
    └── MODEL_NAME__SANITIZED/
        ├── results.json
        └── samples/
```

Where:
- `TASK_NAME` is the name of the evaluation task
- `MODEL_NAME__SANITIZED` is the model name with slashes replaced by double underscores
- `results.json` contains the evaluation metrics
- `samples/` directory contains detailed sample-level results

## Common Tasks

Here are some commonly used evaluation tasks:

### Language Understanding
- `hellaswag`: Commonsense reasoning
- `piqa`: Physical interaction reasoning
- `winogrande`: Winograd schema challenge

### Mathematical Reasoning
- `gsm8k`: Grade school math problems
- `gsm8k_cot`: GSM8K with chain-of-thought
- `math`: Mathematical problem solving

### Reading Comprehension
- `lambada_openai`: Language modeling evaluation
- `arc_easy`: Science questions (easy)
- `arc_challenge`: Science questions (challenging)

### Code Understanding
- `humaneval`: Code generation evaluation
- `mbpp`: Python programming problems
