# clip-vit-base-patch32
for tv_prune_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.999 1.0; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=1 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=false \
        fast_dev_run=false \
        method.save_checkpoint=outputs/sparse_we_moe/clip-vit-base-patch32/clip-vit-base-patch32_TA8_sparse_we_moe_checkpoint_${prune_ratio_name}.ckpt \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path=outputs/sparse_we_moe/clip-vit-base-patch32/sparse_we_moe_ratio_${prune_ratio_name}.json
done

# clip-vit-base-patch32 (1-LAYER Router)
for tv_prune_ratio in 0.9 0.99; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=1 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=false \
        method.router_hidden_layers=1 \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path=outputs/sparse_we_moe/clip-vit-base-patch32/sparse_we_moe_ratio_${prune_ratio_name}_one_layer_router.json
done

# clip-vit-base-patch16
for tv_prune_ratio in 1.0 0.999 0.99 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=3 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=false \
        fast_dev_run=false \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
        report_save_path=outputs/sparse_we_moe/clip-vit-base-patch16/sparse_we_moe_ratio_${prune_ratio_name}.json
done

# clip-vit-large-patch14
for tv_prune_ratio in 1.0 0.999 0.99 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do
    prune_ratio_name=$(echo $tv_prune_ratio | tr '.' '_')
    CUDA_VISIBLE_DEVICES=3 fusion_bench \
        method=wemoe/sparse_weight_ensembling_moe \
        method.name=sparse_clip_weight_ensembling_moe \
        method.tv_prune_ratio=$tv_prune_ratio \
        method.use_grad_accumulate=true \
        fast_dev_run=false \
        method.batch_size=4 \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
        report_save_path=outputs/sparse_we_moe/clip-vit-large-patch14/sparse_we_moe_ratio_${prune_ratio_name}.json
done
