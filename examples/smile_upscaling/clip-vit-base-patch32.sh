#! /bin/bash
# This script is used to run the experiments using CLIP-ViT-B-32 models

function run_eight_tasks() {
    fusion_bench \
        method=smile_upscaling/smile_upscaling \
        method.device=cuda \
        method.gate_k=$gate_k method.k=$k \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path="outputs/ViT-B-32/eight_tasks/gate_k\=${gate_k}_k\=${k}.json"
}

function vit_b32_run_20_tasks() {
    # simple average
    report_path="outputs/ViT-B-32/20_tasks/simple_average.json"
    if [ ! -f $report_path ]; then
        rich-run fusion_bench \
            method=simple_average \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
            taskpool.base_model=openai/clip-vit-base-patch32 \
            report_save_path=$report_path
    else
        echo "Report already exists at $report_path"
    fi

    # task arithmetic
    report_path="outputs/ViT-B-32/20_tasks/task_arithmetic.json"
    if [ ! -f $report_path ]; then
        rich-run fusion_bench \
            method=task_arithmetic \
            method.scaling_factor=0.1 \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
            taskpool.base_model=openai/clip-vit-base-patch32 \
            report_save_path=$report_path
    else
        echo "Report already exists at $report_path"
    fi

    # ties-merging
    report_path="outputs/ViT-B-32/20_tasks/ties_merging.json"
    if [ ! -f $report_path ]; then
        rich-run fusion_bench \
            method=ties_merging \
            method.scaling_factor=0.1 \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
            taskpool.base_model=openai/clip-vit-base-patch32 \
            report_save_path=$report_path
    else
        echo "Report already exists at $report_path"
    fi

    # regmean
    report_path="outputs/ViT-B-32/20_tasks/regmean.json"
    if [ ! -f $report_path ]; then
        rich-run fusion_bench \
            method=regmean/clip_regmean \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
            taskpool.base_model=openai/clip-vit-base-patch32 \
            report_save_path=$report_path
    else
        echo "Report already exists at $report_path"
    fi

    # fisher-merging
    report_path="outputs/ViT-B-32/20_tasks/fisher.json"
    if [ ! -f $report_path ]; then
        rich-run fusion_bench \
            method=fisher_merging/clip_fisher_merging \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
            taskpool.base_model=openai/clip-vit-base-patch32 \
            report_save_path=$report_path
    else
        echo "Report already exists at $report_path"
    fi

    # TSVM
    report_path="outputs/ViT-B-32/20_tasks/tsvm.json"
    if [ ! -f $report_path ]; then
        rich-run fusion_bench \
            method=task_singular_vector/TaskSingularVectorMerging \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
            taskpool.base_model=openai/clip-vit-base-patch32 \
            report_save_path=$report_path
    else
        echo "Report already exists at $report_path"
    fi

    # smile upscaling
    report_path="outputs/ViT-B-32/20_tasks/gate_k=${gate_k}_k=${k}.json"
    if [ ! -f $report_path ]; then
        rich-run fusion_bench \
            method=smile_upscaling/smile_upscaling \
            method.device=cuda \
            method.gate_k=$gate_k method.k=$k \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
            taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
            taskpool.base_model=openai/clip-vit-base-patch32 \
            report_save_path="outputs/ViT-B-32/20_tasks/gate_k\=${gate_k}_k\=${k}.json"
    else
        echo "Report already exists at $report_path"
    fi
}

# run the generalization experiment
function run_generalization_exp() {
    fusion_bench \
        method=smile_upscaling/smile_upscaling \
        method.device=cuda \
        method.gate_k=$gate_k method.k=$k \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        report_save_path="outputs/ViT-B-32/generalization_exp/gate_k\=${gate_k}_k\=${k}.json"
}

function run_generalization_exp_task_arithmetic() {
    fusion_bench \
        method=task_arithmetic \
        method.scaling_factor=0.3 \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        report_save_path="outputs/ViT-B-32/generalization_exp_task_arithmetic/result.json"
}

function run_generalization_exp_ties_merging() {
    fusion_bench \
        method=ties_merging \
        method.scaling_factor=0.3 \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        report_save_path="outputs/ViT-B-32/generalization_exp_ties_merging/result.json"
}

function run_generalization_exp_fisher_merging() {
    fusion_bench \
        method=fisher_merging/clip_fisher_merging \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        report_save_path="outputs/ViT-B-32/generalization_exp_fisher_merging/result.json"
}

function run_generalization_exp_regmean() {
    fusion_bench \
        method=regmean/clip_regmean \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
        report_save_path="outputs/ViT-B-32/generalization_exp_regmean/result.json"
}

# Example Usage
gate_k=16 k=32 run_eight_tasks
gate_k=16 k=128 run_eight_tasks

# hyper-parameter search
for gate_k in 1 2 4 8 16 32 64 128 256 512 768; do
    for k in 4 8 16 32 64 128 -1; do
        run_eight_tasks
    done
done

# Routing Analysis: save layer-wise routing weights
function run_routing_analysis() {
    SAVE_PATH="outputs/ViT-B-32/routing_analysis/gate_k\=${gate_k}_k\=${k}"
    fusion_bench \
        method=smile_upscaling/smile_upscaling \
        method.device=cuda \
        method.gate_k=$gate_k method.k=$k \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool._target_=fusion_bench.taskpool.clip_vision.SmileCLIPVisionModelTaskPool \
        +taskpool.linear_module_names="[mlp.fc1, mlp.fc2]" \
        +taskpool.layer_wise_routing_weights_save_path=${SAVE_PATH} \
        +taskpool.layer_wise_routing_weights_max_num=1000 \
        report_save_path=${SAVE_PATH}/result.json
}

gate_k=16 k=32 run_routing_analysis

gate_k=16 k=128 vit_b32_run_20_tasks
