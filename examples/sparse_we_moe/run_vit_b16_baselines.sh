# clip-vit-base-patch16

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    fast_dev_run=false \
    method=wemoe/weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.use_grad_accumulate=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-weight_ensembling_moe.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=adamerging/clip \
    method.name=clip_layer_wise_adamerging \
    fast_dev_run=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_layer_wise_adamerging.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=ties_merging \
    method.scaling_factor=0.3 method.threshold=20 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_ties_merging.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=task_arithmetic \
    method.scaling_factor=0.3 modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_task_arithmetic.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=simple_average \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-simple_average.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=regmean/clip_regmean \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-clip_regmean.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=fisher_merging/clip_fisher_merging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-fisher_merging.json

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    modelpool.models.0.path=openai/clip-vit-base-patch16 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
    report_save_path=outputs/clip-vit-base-patch16/clip-vit-base-patch16-pretrain.json
