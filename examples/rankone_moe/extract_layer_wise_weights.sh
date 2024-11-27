# extract layer-wise routing weights of the rankone-wemoe model

task_num=8
rank_k=16
select_k_factor=1
select_k=$(printf "%.0f" $(echo "$rank_k * $task_num * $select_k_factor" | bc))
rank_k_name=$(echo $rank_k | tr '.' '_')
select_k_factor_name=$(echo $select_k_factor | tr '.' '_')

CUDA_VISIBLE_DEVICES=1 fusion_bench \
    method=rankone_wemoe/rankone_wemoe \
    method.name=rankone_wemoe \
    method.rank_k=$rank_k \
    method.select_k=$select_k \
    method.save_checkpoint=outputs/rankone_wemoe/clip-vit-base-patch32/layer_wise_routing_weights/rankone_wemoe_rank_${rank_k_name}_select_factor_${select_k_factor_name}.ckpt \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip_rankone_wemoe_clip-vit-classification_TA8 \
    taskpool.dataloader_kwargs.shuffle=true \
    taskpool.layer_wise_routing_weights_save_path=outputs/rankone_wemoe/clip-vit-base-patch32/layer_wise_routing_weights/ \
    taskpool.layer_wise_routing_weights_max_num=1000
