# clip-vit-base-patch32
for tv_prune_ratio in 0.9; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=1 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=false \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_clean \
        taskpool=CLIPVisionModelTaskPool/clip-vit-base-patch32_robustness_clean \
        save_report=outputs/sparse_we_moe/robustness/clip-vit-base-patch32/sparse_we_moe_ratio_${prune_ratio_name}_corruption_clean.json
done

for tv_prune_ratio in 0.9; do
    for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
        prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
        CUDA_VISIBLE_DEVICES=1 fusion_bench \
            method=wemoe/sparse_weight_ensembling_moe \
            method.name=sparse_clip_weight_ensembling_moe \
            method.tv_prune_ratio=$tv_prune_ratio \
            method.use_grad_accumulate=false \
            fast_dev_run=false \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
            modelpool.corruption=${corruption} \
            taskpool=CLIPVisionModelTaskPool/clip-vit-base-patch32_robustness_corrupted \
            taskpool.corruption=${corruption} \
            save_report=outputs/sparse_we_moe/robustness/clip-vit-base-patch32/sparse_we_moe_ratio_${prune_ratio_name}_corruption_${corruption}.json
    done
done

# clip-vit-base-patch16
for tv_prune_ratio in 0.9; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=3 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=false \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_robustness_clean \
        taskpool=CLIPVisionModelTaskPool/clip-vit-base-patch16_robustness_clean \
        report_save_path=outputs/sparse_we_moe/robustness/clip-vit-base-patch16/sparse_we_moe_ratio_${prune_ratio_name}_corruption_clean.json
done

for tv_prune_ratio in 0.9; do
    for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter; do
        prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
        CUDA_VISIBLE_DEVICES=3 fusion_bench \
            method=wemoe/sparse_weight_ensembling_moe \
            method.name=sparse_clip_weight_ensembling_moe \
            method.tv_prune_ratio=$tv_prune_ratio \
            method.use_grad_accumulate=false \
            fast_dev_run=false \
            modelpool=CLIPVisionModelPool/clip-vit-base-patch16_robustness_corrupted \
            modelpool.corruption=${corruption} \
            taskpool=CLIPVisionModelTaskPool/clip-vit-base-patch16_robustness_corrupted \
            taskpool.corruption=${corruption} \
            report_save_path=outputs/sparse_we_moe/robustness/clip-vit-base-patch16/sparse_we_moe_ratio_${prune_ratio_name}_corruption_${corruption}.json
    done
done
