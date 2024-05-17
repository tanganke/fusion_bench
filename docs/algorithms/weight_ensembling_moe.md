# Weight-Ensenbling Mixture of Experts

<figure markdown="span">
    ![alt text](images/wemoe.png){ width="90%" }
    <figcaption style="max-width:90%">
(a) **Framework overview**. This figure shows the overall framework of our proposed method to merge the pre-trained model and fine-tuned task-specific models. We merge weights in the Transformer Layers except for the MLPs. For the MLPs, we upcycle them into weight-assembling MoE modules.
(b) **Wieght-Ensembling Mixture of Experts (MoE) Module**. Here we outline the detailed structure of the Weight-Ensembling MoE module, composed of the router, pre-trained MLP weights, and a collection of task vectors. Collaboration between shared weights and task vectors is employed to create input-conditioned weights dynamically. In this way, we separate shared information and task-specific knowledge, which are then combined based on input in time.
    </figcaption>
</figure>

This method is designed to handle a wide range of tasks by segregating shared information and task-specific knowledge. 
It dynamically combines these elements based on the input samples.

The Weight-Ensembling MoE module consists of three main components: the router, the pre-trained MLP weights, and a collection of task vectors. 
The router, which is an MLP, processes the input data and generates routing weights. These weights determine how the knowledge from different tasks is combined.
The pre-trained MLP weights are crucial as they have been trained to recognize a wide range of data patterns. 
The task vectors represent the differences between the MLPs that have been fine-tuned for specific tasks and the pre-trained ones, capturing the unique adjustments made to optimize them for specific tasks.
The routing weights are averaged across the input tokens, and these weights are used to select task vectors from a dictionary matrix.
These task vectors are then added to the pre-trained MLP weights to create input-conditioned weights.


## Code Integration

multi-task model fusion experiment on eight image classification tasks.

```bash
# merge eight CLIP-ViT-B/32 models using WE MoE
fusion_bench \
  method=weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.use_grad_accumulate=false \
    method.save_checkpoint=outputs/clip-vit-base-patch32_TA8_weight_ensembling_moe_checkpoint.ckpt \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8
```

merge eight CLIP-ViT-L/14 models:

```bash
# merge eight CLIP-ViT-L/14 models using WE MoE, fine-tune the routers
fusion_bench print_config=false \
  method=weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.use_grad_accumulate=true \
    method.save_checkpoint=outputs/clip-vit-large-patch14_TA8_weight_ensembling_moe_checkpoint.ckpt \
    method.batch_size=4 method.devices=4 \
  modelpool=clip-vit-large-patch14_TA8 \
  taskpool=dummy &&

# load the checkpoint and evaluate the model
fusion_bench \
  method=weight_ensembling_moe \
    method.name=clip_weight_ensembling_moe \
    method.checkpoint=outputs/clip-vit-large-patch14_TA8_weight_ensembling_moe_checkpoint.ckpt \
  modelpool=clip-vit-large-patch14_TA8 \
  taskpool=clip-vit-classification_TA8 \
    taskpool.clip_model: openai/clip-vit-large-patch14
```


[^1]: Merging Multi-Task Models via Weight-Ensembling Mixture of Experts. http://arxiv.org/abs/2402.00433
