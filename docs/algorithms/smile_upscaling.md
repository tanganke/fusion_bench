# SMILE Upscaling

## Examples

### CLIP-ViT-B/32 on eight tasks

Evaluate single fine-tuned models and save the results to `outputs/ViT-B-32/single-task/` and `outputs/ViT-L-14/single-task/` for CLIP-ViT-B/32 and CLIP-ViT-L/14 models, respectively.

```bash
# evaluate singlue fine-tuned models
for task in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd
do
    fusion_bench method=dummy \
        modelpool=clip-vit-base-patch32_individual \
            modelpool.models.0.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8 \
        save_report="outputs/ViT-B-32/single-task/clip-vit-base-patch32_${task}.json"
done

# if you have multiple GPUs, you can run the following code to evaluate the CLIP-ViT-L/14 models in parallel
# evaluate singlue fine-tuned models clip-vit-large
tasks=(sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd)
CUDA_DEVICES=(0 1 2 3 4 5 6 7)  # List of CUDA devices to use

for i in "${!CUDA_DEVICES[@]}"; do
    task=${tasks[$i]}
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$i]} fusion_bench method=dummy \
        modelpool=clip-vit-large-patch14_individual \
            modelpool.models.0.path=tanganke/clip-vit-large-patch14_${task} \
        taskpool=clip-vit-classification_TA8 \
            taskpool.clip_model=openai/clip-vit-large-patch14 \
        save_report="outputs/ViT-L-14/single-task/clip-vit-large-patch14_${task}.json" &
done
```

Upscale eight CLIP-ViT-B/32 models with SMILE, each CLIP-ViT-B/32 model is trained on a downstream task.

```bash
gate_k=16
k=32
fusion_bench \
    method=smile_upscaling \
        method.device=cuda \
        method.gate_k=$gate_k method.k=$k \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8.local \
    save_report="outputs/ViT-B-32/eight_tasks/gate_k\=${gate_k}_k\=${k}.json"
```

Hyperparameter search for SMILE upscaling.

```bash
for gate_k in 1 2 4 8 16 32 64 128 256 512 768; do
    for k in 4 8 16 32 64 128 -1; do
        fusion_bench \
            method=smile_upscaling \
                method.device=cuda \
                method.gate_k=$gate_k method.k=$k \
            modelpool=clip-vit-base-patch32_TA8 \
            taskpool=clip-vit-classification_TA8.local \
            save_report="outputs/ViT-B-32/eight_tasks/gate_k\=${gate_k}_k\=${k}.json"
    done
done
```

Ablations on number of experts per token (Top-K).

```bash
gate_k=16
k=32
for top_k in 1 2 4
do
fusion_bench \
    method=smile_upscaling \
        method.device=cuda \
        method.gate_k=$gate_k method.k=$k \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8.local \
    save_report="outputs/ViT-B-32/ablation/gate_k\=${gate_k}_k\=${k}.json"
done
```

### CLIP-ViT-L/14 on eight tasks

hyperparameter search for SMILE upscaling.

```bash
for gate_k in 1 2 4 8 16 32 64 128; do
    for k in 4 8 16 32 64 128 -1; do
        fusion_bench \
            method=smile_upscaling \
                method.gate_k=$gate_k method.k=$k \
            modelpool=clip-vit-large-patch14_TA8 \
            taskpool=clip-vit-classification_TA8.local \
                taskpool.clip_model=openai/clip-vit-large-patch14 \
            save_report="outputs/ViT-B-32/eight_tasks/gate_k\=${gate_k}_k\=${k}.json"
    done
done
```

### Flan-T5 models on eight tasks from GLUE benchmark

Hyperparameter search for full fine-tuned and lora fine-tuned Flan-T5 models.

```bash
# hyperparameter search for full fine-tuned flan-t5-base
for gate_k in 4 8 16 32; do
    for k in 16 32 64 128; do
        fusion_bench \
            method=smile_upscaling \
                method.device=cpu \
                method.gate_k=$gate_k method.k=$k \
            modelpool=flan-t5-base_glue \
            taskpool=flan-t5_glue_text_generation \
            save_report="outputs/flan-t5-base/glue_text_generation/gate_k\=${gate_k}_k\=${k}.json"
    done
done

# hyperparameter search for lora fine-tuned flan-t5-base
for gate_k in 2 4 8; do
    for k in 4 8 16; do
        fusion_bench \
            method=smile_upscaling \
                method.device=cuda \
                method.gate_k=$gate_k method.k=$k \
            modelpool=flan-t5-base_glue_lora16 \
            taskpool=flan-t5_glue_text_generation \
            save_report="outputs/flan-t5-base_lora16/glue_text_generation/gate_k\=${gate_k}_k\=${k}.json"
    done
done
```

### Upscale Mistral-7B models

1. Prepare the following 4 configuration files in `configs/modelpool`:

    ``` yaml title="config/modelpool/smile_mistral_exp_v1.yaml"
    type: AutoModelForCausalLMPool
    models:
    - name: _pretrained_
        path: /data/huggingface_models/mistralai/Mistral-7B-v0.1
    - name: expert_1
        path: /data/huggingface_models/meta-math/MetaMath-Mistral-7B

    dtype: float16
    ```

    ```yaml title="config/modelpool/smile_mistral_exp_v2.yaml"
    type: AutoModelForCausalLMPool
    models:
    - name: _pretrained_
        path: /data/huggingface_models/mistralai/Mistral-7B-v0.1
    - name: expert_1
        path: /data/huggingface_models/cognitivecomputations/dolphin-2.1-mistral-7b

    dtype: float16
    ```

    ```yaml title="config/modelpool/smile_mistral_exp_v3.yaml"
    type: AutoModelForCausalLMPool
    models:
    - name: _pretrained_
        path: /data/huggingface_models/mistralai/Mistral-7B-v0.1
    - name: expert_1
        path: /data/huggingface_models/uukuguy/speechless-code-mistral-7b-v1.0

    dtype: float16
    ```

    ```yaml title="config/modelpool/smile_mistral_exp_v4.yaml"
    type: AutoModelForCausalLMPool
    models:
    - name: _pretrained_
        path: /data/huggingface_models/mistralai/Mistral-7B-v0.1
    - name: expert_1
        path: /data/huggingface_models/meta-math/MetaMath-Mistral-7B
    - name: expert_2
        path: /data/huggingface_models/cognitivecomputations/dolphin-2.1-mistral-7b
    - name: expert_3
        path: /data/huggingface_models/uukuguy/speechless-code-mistral-7b-v1.0

    dtype: float16
    ```

2. Upscale Mistral-7B models.

    ```bash
    function model_fusion() {
        output_dir=outputs/llama/gate_k-${gate_k}_k-${k}/version_${version}
        fusion_bench \
            method=smile_mistral_upscaling \
                method.rank_of_router=$gate_k method.rank_of_expert=$k \
                method.model_path=${output_dir} \
            modelpool=smile_mistral_exp_v${version} \
                modelpool.dtype=float32 \
            taskpool=dummy \
            save_report="${output_dir}/model_info.json"
    }

    gate_k=8
    for k in 8 16 32 64 128 256 384 512; do
        for version in 1 2 3 4; do
            model_fusion
        done
    done
    ```

3. Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/) to evaluate the models. We use the default configurations for each task.

    ```bash
    # For some GPUs, the following environment variables need to be set
    # export NCCL_P2P_DISABLE="1"
    # export NCCL_IB_DISABLE="1"

    function model_eval() {
        output_dir=outputs/llama/test/gate_k-${gate_k}_k-${k}/version_${version}

        # Check if ${output_dir}/${task}.json exists as a directory and return if it does
        if [ -d "${output_dir}/${task}.json" ]; then
            echo "Directory ${output_dir}/${task}.json already exists. Skipping evaluation."
            return
        fi

        lm_eval --model hf \
            --model_args pretrained=${output_dir},dtype="float16",parallelize=True \
            --tasks ${task} \
            --output_path ${output_dir}/${task}.json \
            --batch_size 6
    }
    ```

    The above function can be used to evaluate the models on specified task.

    ```bash
    # Evaluate all the models on GSM8K task
    gate_k=8
    task=gsm8k
    for k in 8 16 32 64 128 256 384 512; do
        for version in 1 2 3 4; do
            model_eval
        done
    done

    # Evaluate all M0;123 models on truthfulqa gsm8k arc_challenge mmlu
    k=8
    version=4
    for task in truthfulqa gsm8k arc_challenge mmlu; do
        model_eval
    done
    ```

    The reported metrics are:

    - mmlu (general): acc
    - truthfulqa (truthful): mc2
    - gsm8k (math): flexible exact match
    - arc_challenge (reasoning): acc_norm


## Scope

### Projection Merge Experiments

```bash
# project into different subspaces
for task in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd
do
    # Space I
    CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=singular_projection_merging \
            method.device=cuda method.rank=low method.k=-1 method.full_matrices=false \
        modelpool=clip-vit-base-patch32_single_finetuned \
            modelpool.models.1.name=${task} \
            modelpool.models.1.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8.local \
        save_report="outputs/ViT-B-32/single-task/projection_merging_zone1_${task}.json" &

    # Space II
    CUDA_VISIBLE_DEVICES=1 fusion_bench \
        method=singular_projection_merging \
            method.device=cuda method.rank=high method.k=-1 method.full_matrices=false \
        modelpool=clip-vit-base-patch32_single_finetuned \
            modelpool.models.1.name=${task} \
            modelpool.models.1.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8.local \
        save_report="outputs/ViT-B-32/single-task/projection_merging_zone2_${task}.json" &

    # Space III
    CUDA_VISIBLE_DEVICES=2 fusion_bench \
        method=singular_projection_merging \
            method.device=cuda method.rank=high method.k=-1 method.full_matrices=true \
        modelpool=clip-vit-base-patch32_single_finetuned \
            modelpool.models.1.name=${task} \
            modelpool.models.1.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8.local \
        save_report="outputs/ViT-B-32/single-task/projection_merging_zone23_${task}.json" &
    wait
done
```
