fusion_bench \
    method=dummy \
    modelpool=OpenCLIPVisionModelPool/ViT-B-32_individual \
    taskpool=OpenCLIPVisionModelTaskPool/ViT-B-32_TA8

fusion_bench \
    method=dummy \
    modelpool=OpenCLIPVisionModelPool/ViT-B-32_individual \
    modelpool.models._pretrained_.pickle_path="$\{...model_dir\}/ViT-B-32/SUN397/finetuned.pt" \
    taskpool=OpenCLIPVisionModelTaskPool/ViT-B-32_TA8
