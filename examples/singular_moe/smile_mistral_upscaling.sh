function model_fusion() {
    output_dir=outputs/llama/test/gate_k-${gate_k}_k-${k}/version_${version}
    fusion_bench \
        method=smile_mistral_upscaling \
        method.rank_of_router=$gate_k method.rank_of_expert=$k \
        method.model_path=${output_dir} \
        modelpool=llama_merging_v${version} \
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

function model_eval() {
    output_dir=outputs/llama/test/gate_k-${gate_k}_k-${k}/version_${version}

    # Check if ${output_dir}/${task}.json exists as a directory and return if it does
    if [ -d "${output_dir}/${task}.json" ]; then
        echo "Directory ${output_dir}/${task}.json already exists. Skipping evaluation."
        return
    fi

    export NCCL_P2P_DISABLE="1"
    export NCCL_IB_DISABLE="1"
    lm_eval --model fusion_bench \
        --model_args pretrained=${output_dir},dtype="float16",parallelize=True \
        --tasks ${task} \
        --output_path ${output_dir}/${task}.json \
        --batch_size 6
}

gate_k=8
for k in 8 16 32 64 128 256 384 512; do
    for version in 1 2 3 4; do
        for task in truthfulqa gsm8k arc_challenge mmlu; do
            model_eval
        done
    done
done
