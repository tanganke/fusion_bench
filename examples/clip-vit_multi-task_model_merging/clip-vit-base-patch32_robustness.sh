# Task Arithmetic
fusion_bench \
    method=task_arithmetic \
        method.scaling_factor=0.3 \
    modelpool=clip-vit-base-patch32_robustness_clean \
    taskpool=clip-vit-base-patch32_robustness_clean \
    report_save_path=outputs/clip-vit-base-patch32_robustness_clean_task_arithmetic.json

for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
do
    fusion_bench --config-name clip-vit-base-patch32_robustness_corrupted \
        corruption=$corruption \
        method=task_arithmetic \
            method.scaling_factor=0.3 \
        report_save_path=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_task_arithmetic.json
done

# Ties-Merging
fusion_bench \
    method=ties_merging \
        method.scaling_factor=0.3 \
    modelpool=clip-vit-base-patch32_robustness_clean \
    taskpool=clip-vit-base-patch32_robustness_clean \
    report_save_path=outputs/clip-vit-base-patch32_robustness_clean_ties_merging.json

for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
do
    fusion_bench --config-name clip-vit-base-patch32_robustness_corrupted \
        corruption=$corruption \
        method=ties_merging \
            method.scaling_factor=0.3 \
        report_save_path=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_ties_merging.json
done

# Fisher Merging
fusion_bench \
    method=clip_fisher_merging \
    modelpool=clip-vit-base-patch32_robustness_clean \
    taskpool=clip-vit-base-patch32_robustness_clean \
    report_save_path=outputs/clip-vit-base-patch32_robustness_clean_clip_fisher_merging.json

for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
do
    fusion_bench --config-name clip-vit-base-patch32_robustness_corrupted \
        corruption=$corruption \
        method=clip_fisher_merging \
        report_save_path=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_clip_fisher_merging.json
done

# RegMean
fusion_bench \
    method=clip_regmean \
    modelpool=clip-vit-base-patch32_robustness_clean \
    taskpool=clip-vit-base-patch32_robustness_clean \
    report_save_path=outputs/clip-vit-base-patch32_robustness_clean_clip_regmean.json

for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
do
    fusion_bench --config-name clip-vit-base-patch32_robustness_corrupted \
        corruption=$corruption \
        method=clip_regmean \
        report_save_path=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_clip_regmean.json
done

# AdaMerging
fusion_bench \
    method=adamerging/clip \
        method.name=clip_layer_wise_adamerging \
        method.save_merging_weights=outputs/clip-vit-base-patch32_robustness_clean_layer_wise_adamerging_weights.pt \
    modelpool=clip-vit-base-patch32_robustness_clean \
    taskpool=clip-vit-base-patch32_robustness_clean \
    report_save_path=outputs/clip-vit-base-patch32_robustness_clean_clip_layer_wise_adamerging.json

for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
do
    fusion_bench --config-name clip-vit-base-patch32_robustness_corrupted \
        corruption=$corruption \
        method=adamerging/clip \
            method.name=clip_layer_wise_adamerging \
            method.save_merging_weights=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_layer_wise_adamerging_weights.pt \
        report_save_path=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_clip_layer_wise_adamerging.json
done

# WEMOE
fusion_bench \
    method=weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
        method.use_grad_accumulate=false \
        method.save_checkpoint=outputs/clip-vit-base-patch32_robustness_clean_weight_ensembling_moe_checkpoint.ckpt \
    modelpool=clip-vit-base-patch32_robustness_clean \
    taskpool=clip-vit-base-patch32_robustness_clean \
    report_save_path=outputs/clip-vit-base-patch32_robustness_clean_clip_weight_ensembling_moe.json

for corruption in contrast gaussian_noise impulse_noise jpeg_compression motion_blur pixelate spatter
do
    fusion_bench --config-name clip-vit-base-patch32_robustness_corrupted \
        corruption=$corruption \
        method=weight_ensembling_moe \
            method.name=clip_weight_ensembling_moe \
            method.lr=5e-5 \
            method.use_grad_accumulate=false \
            method.save_checkpoint=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_weight_ensembling_moe_checkpoint.ckpt \
        report_save_path=outputs/clip-vit-base-patch32_robustness_corrupted_${corruption}_clip_weight_ensembling_moe.json
done
