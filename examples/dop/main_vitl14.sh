cuda=1
seeds=(40 41 42 43 44 45 46 47 48 49)

# 8 tasks
for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    fabric.loggers.root_dir=outputs/dop/dop/vit-l-14-TA8/ \
    fabric.loggers.name='main' \
    fabric.loggers.version=${version} \
    method=dop/dop \
    method.mgda=true \
    method.seed=${seeds[version]} \
    method.ema=true \
    method.ema_beta=0.999 \
    method.alpha=0.8 \
    method.evaluate_on_every_step=true \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14   || true
done

# 14 tasks
for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    fabric.loggers.root_dir=outputs/dop/dop/vit-l-14-TA14/ \
    fabric.loggers.name='main' \
    fabric.loggers.version=${version} \
    method=dop/dop \
    method.mgda=true \
    method.seed=${seeds[version]} \
    method.ema=true \
    method.ema_beta=0.999 \
    method.alpha=0.8 \
    method.evaluate_on_every_step=true \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL14_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14_L14 || true
done

# 20 tasks
for version in {0..9}; do
    CUDA_VISIBLE_DEVICES=$cuda fusion_bench \
    fabric.loggers.root_dir=outputs/dop/dop/vit-l-14-TA20/ \
    fabric.loggers.name='main' \
    fabric.loggers.version=${version} \
    method=dop/dop \
    method.mgda=true \
    method.seed=${seeds[version]} \
    method.ema=true \
    method.ema_beta=0.999 \
    method.alpha=0.8 \
    method.evaluate_on_every_step=true \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TALL20_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20_L14 || true
done