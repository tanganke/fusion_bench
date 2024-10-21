# clip-vit-base-patch32

# method.batch_reduce=true
for tv_prune_ratio in 0.0 0.9; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=1 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=false \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_generalization_exp1 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path=outputs/sparse_we_moe/generalization_exp1/sparse_we_moe_ratio_${prune_ratio_name}.json
done

# method.batch_reduce=false
for tv_prune_ratio in 0.0 0.9; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=1 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=false \
        method.batch_reduce=false \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_generalization_exp1 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path=outputs/sparse_we_moe/generalization_exp1/sparse_we_moe_ratio_${prune_ratio_name}_nonbatch_reduce.json
done
