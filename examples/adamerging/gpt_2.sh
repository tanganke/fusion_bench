# Layer-wise AdaMerigng for GPT-2
fusion_bench \
    method=adamerging/layer_wise_gpt2 \
    method.max_steps=400 \
    modelpool=test/test.yaml \
    taskpool=test/test.yaml

