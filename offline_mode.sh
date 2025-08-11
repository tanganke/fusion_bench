#!/bin/bash
# This script is used to set the environment variables for offline mode
# Usage:
# source offline_mode.sh [disable]
# If 'disable' is provided as an argument, offline mode will be disabled

if [ "$1" = "disable" ]; then
    unset FUSION_BENCH_OFFLINE
    unset TRANSFORMERS_OFFLINE
    unset HF_DATASETS_OFFLINE
    echo "Offline mode disabled"
else
    export FUSION_BENCH_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "Offline mode enabled. You can disable it by running: 'source offline_mode.sh disable'"
fi
