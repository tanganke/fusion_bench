# clip-vit-base-patch32
task_num=8
for rank_k in 16 32 64 128 256 512; do
  for select_k_factor in 0.25 0.5 0.75 1; do
    select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
    rank_k_name=$(echo $rank_k | tr '.' '_') || true
    select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
    CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=rankone_moe/rankone_moe \
        method.name=rankone_moe \
        method.rank_k=$rank_k \
        method.select_k=$select_k \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path=outputs/rankone_moe/clip-vit-base-patch32/rankone_moe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done

# clip-vit-base-patch16
task_num=8
for rank_k in 16 32 64 128 256 512; do
  for select_k_factor in 0.25 0.5 0.75 1; do
    select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
    rank_k_name=$(echo $rank_k | tr '.' '_') || true
    select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
    CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=rankone_moe/rankone_moe \
        method.name=rankone_moe \
        method.rank_k=$rank_k \
        method.select_k=$select_k \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
        report_save_path=outputs/rankone_moe/clip-vit-base-patch16/rankone_moe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done

# clip-vit-large-patch14
task_num=8
for rank_k in 16 32 64 128 256 512; do
  for select_k_factor in 0.25 0.5 0.75 1; do
    select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc)) || true
    rank_k_name=$(echo $rank_k | tr '.' '_') || true
    select_k_factor_name=$(echo $select_k_factor | tr '.' '_') || true
    CUDA_VISIBLE_DEVICES=0 fusion_bench \
        method=rankone_moe/rankone_moe \
        method.name=rankone_moe \
        method.rank_k=$rank_k \
        method.select_k=$select_k \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
        report_save_path=outputs/rankone_moe/clip-vit-large-patch14/rankone_moe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.json || true
  done
done