# Taks Arithmetic
fusion_bench \
    method=task_arithmetic \
        method.scaling_factor=0.3 \
    modelpool=clip-vit-base-patch32_generalization_exp1 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp1_task_arithmetic.json

fusion_bench \
    method=task_arithmetic \
        method.scaling_factor=0.3 \
    modelpool=clip-vit-base-patch32_generalization_exp2 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp2_task_arithmetic.json

# Ties-Merging
fusion_bench \
    method=ties_merging \
        method.scaling_factor=0.3 method.threshold=20 \
    modelpool=clip-vit-base-patch32_generalization_exp1 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp1_ties_merging.json

fusion_bench \
    method=ties_merging \
        method.scaling_factor=0.3 method.threshold=20 \
    modelpool=clip-vit-base-patch32_generalization_exp2 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp2_ties_merging.json

# Fisher Merging
fusion_bench method=clip_fisher_merging \
    modelpool=clip-vit-base-patch32_generalization_exp1 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp1_clip_fisher_merging.json

fusion_bench method=clip_fisher_merging \
    modelpool=clip-vit-base-patch32_generalization_exp2 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp2_clip_fisher_merging.json

# RegMean
fusion_bench method=clip_regmean \
    modelpool=clip-vit-base-patch32_generalization_exp1 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp1_clip_regmean.json

fusion_bench method=clip_regmean \
    modelpool=clip-vit-base-patch32_generalization_exp2 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp2_clip_regmean.json


# AdaMerging
fusion_bench \
    method=adamerging/clip \
    method.name=clip_layer_wise_adamerging \
    method.save_merging_weights=outputs/clip-vit-base-patch32_generalization_exp1_layer_wise_adamerging_weights.pt \
    modelpool=clip-vit-base-patch32_generalization_exp1 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp1_clip_layer_wise_adamerging.json

fusion_bench \
    method=adamerging/clip \
    method.name=clip_layer_wise_adamerging \
    method.save_merging_weights=outputs/clip-vit-base-patch32_generalization_exp2_layer_wise_adamerging_weights.pt \
    modelpool=clip-vit-base-patch32_generalization_exp2 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp2_clip_layer_wise_adamerging.json

# WEMOE
fusion_bench \
    method=weight_ensembling_moe \
        method.lr=2e-3 \
        method.name=clip_weight_ensembling_moe \
        method.use_grad_accumulate=false \
        method.save_checkpoint=outputs/clip-vit-base-patch32_generalization_exp1_weight_ensembling_moe_checkpoint.ckpt \
    modelpool=clip-vit-base-patch32_generalization_exp1 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp1_clip_weight_ensembling_moe.json

fusion_bench \
    method=weight_ensembling_moe \
        method.lr=2e-3 \
        method.name=clip_weight_ensembling_moe \
        method.use_grad_accumulate=false \
        method.save_checkpoint=outputs/clip-vit-base-patch32_generalization_exp2_weight_ensembling_moe_checkpoint.ckpt \
    modelpool=clip-vit-base-patch32_generalization_exp2 \
    taskpool=clip-vit-classification_TA8 \
    report_save_path=outputs/clip-vit-base-patch32_generalization_exp2_clip_weight_ensembling_moe.json

