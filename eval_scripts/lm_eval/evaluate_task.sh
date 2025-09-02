#!/bin/bash

# Script to evaluate a model on a specific task using lm_eval
# Usage: ./evaluate_task.sh MODEL --tasks TASK --output_path OUTPUT_DIR [--batch_size BATCH_SIZE] [ADDITIONAL_OPTIONS...]

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 MODEL --tasks TASK --output_path OUTPUT_DIR [--batch_size BATCH_SIZE] [--use_vllm] [ADDITIONAL_OPTIONS...]"
    echo ""
    echo "Arguments:"
    echo "  MODEL                        - The model path or name to evaluate (positional argument)"
    echo "  --tasks TASK                 - The task to evaluate on"
    echo "  --output_path OUTPUT_DIR     - Directory to save evaluation results"
    echo "  --batch_size BATCH_SIZE      - Batch size for evaluation (default: auto)"
    echo "  --use_vllm                   - Use vLLM for inference (default: false)"
    echo "  ADDITIONAL_OPTIONS           - Additional arguments for lm_eval (passed through)"
    echo ""
    echo "vLLM Configuration:"
    echo "  When --use_vllm is enabled, the following default vLLM parameters are used:"
    echo "    tensor_parallel_size=1, dtype=auto, gpu_memory_utilization=0.8, data_parallel_size=1"
    echo "  You can override these by passing them as additional model_args via ADDITIONAL_OPTIONS."
    echo ""
    echo "Examples:"
    echo "  # Standard evaluation"
    echo "  $0 'meta-llama/Llama-2-7b-hf' --tasks 'hellaswag' --output_path './results' --batch_size 8 --num_fewshot 5"
    echo ""
    echo "  # Using vLLM for inference with defaults"
    echo "  $0 'meta-llama/Llama-2-7b-hf' --tasks 'lambada_openai' --output_path './results' --use_vllm --batch_size auto"
    echo ""
    echo "  # Using vLLM with custom parameters (passed as additional options)"
    echo "  $0 'meta-llama/Llama-2-7b-hf' --tasks 'lambada_openai' --output_path './results' --use_vllm \\"
    echo "     --model_args 'pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=2,gpu_memory_utilization=0.9'"
    exit 1
}

# Check if help is requested first
for arg in "$@"; do
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        show_usage
    fi
done

# Check if minimum required arguments are provided
if [ $# -lt 1 ]; then
    echo "Error: At least the MODEL argument is required"
    show_usage
fi

# Initialize variables
MODEL=""
TASK=""
OUTPUT_DIR=""
BATCH_SIZE="auto"
USE_VLLM=false
ADDITIONAL_OPTIONS=()

# Get the first positional argument (MODEL)
MODEL="$1"
shift

# Parse arguments manually to handle pass-through options
ADDITIONAL_OPTIONS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASK="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --use_vllm)
            USE_VLLM=true
            shift
            ;;
        *)
            # Any unrecognized option goes to additional options
            ADDITIONAL_OPTIONS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ] || [ -z "$TASK" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "MODEL: '$MODEL'"
    echo "TASK: '$TASK'"
    echo "OUTPUT_DIR: '$OUTPUT_DIR'"
    show_usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Convert model path to directory-safe format (replace / with __)
MODEL_PATH=$(echo "$MODEL" | sed 's/\//__/g')

# Check if output file exists, skip if it does
if [ -d "$OUTPUT_DIR/$TASK/$MODEL_PATH" ]; then
    echo "Skipping $MODEL on $TASK because output file exists at: $OUTPUT_DIR/$TASK/$MODEL_PATH"
    exit 0
else
    echo "Evaluating $MODEL on $TASK"
    echo "Output will be saved to: $OUTPUT_DIR/$TASK/$MODEL_PATH"
    
    # Check if custom model_args are provided in additional options
    CUSTOM_MODEL_ARGS=false
    for option in "${ADDITIONAL_OPTIONS[@]}"; do
        if [[ "$option" == "--model_args" ]]; then
            CUSTOM_MODEL_ARGS=true
            break
        fi
    done

    # Configure model arguments based on inference backend
    if [ "$USE_VLLM" = true ]; then
        echo "Using vLLM for inference"
        MODEL_TYPE="vllm"
        if [ "$CUSTOM_MODEL_ARGS" = false ]; then
            # Use default vLLM model args if no custom ones provided
            MODEL_ARGS="pretrained=$MODEL,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1"
            MODEL_ARGS_FLAG="--model_args"
        else
            # Skip default model args when custom ones are provided
            MODEL_ARGS=""
            MODEL_ARGS_FLAG=""
        fi
    else
        echo "Using standard inference"
        MODEL_TYPE="hf"
        if [ "$CUSTOM_MODEL_ARGS" = false ]; then
            # Use default HF model args if no custom ones provided
            MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,parallelize=True"
            MODEL_ARGS_FLAG="--model_args"
        else
            # Skip default model args when custom ones are provided
            MODEL_ARGS=""
            MODEL_ARGS_FLAG=""
        fi
    fi

    # Run the evaluation
    if [ -n "$MODEL_ARGS" ]; then
        rich-run echo lm_eval \
            --model "$MODEL_TYPE" \
            "$MODEL_ARGS_FLAG" "$MODEL_ARGS" \
            --tasks "$TASK" \
            --output_path "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --confirm_run_unsafe_code \
            "${ADDITIONAL_OPTIONS[@]}"
    else
        rich-run echo lm_eval \
            --model "$MODEL_TYPE" \
            --tasks "$TASK" \
            --output_path "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --confirm_run_unsafe_code \
            "${ADDITIONAL_OPTIONS[@]}"
    fi
    
    echo "Evaluation completed successfully!"
fi
