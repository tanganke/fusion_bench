The structure of upscaled CLIP-ViT-B/32 model is as follows:

```cpp
CLIPVisionModel(                                                                                                           
  (vision_model): CLIPVisionTransformer(
    (embeddings): CLIPVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(32, 32), stride=(32, 32), bias=False)
      (position_embedding): Embedding(50, 768)
    )
    (pre_layrnorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-11): 12 x CLIPEncoderLayer(
          (self_attn): CLIPSdpaAttention(
            (k_proj): SingularMoELinear(in_features=768, out_features=768, num_experts=8, top_k=8, gate_k=16, k=-1)
            (v_proj): SingularMoELinear(in_features=768, out_features=768, num_experts=8, top_k=8, gate_k=16, k=-1)
            (q_proj): SingularMoELinear(in_features=768, out_features=768, num_experts=8, top_k=8, gate_k=16, k=-1)
            (out_proj): SingularMoELinear(in_features=768, out_features=768, num_experts=8, top_k=8, gate_k=16, k=-1)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): SingularMoELinear(in_features=768, out_features=3072, num_experts=8, top_k=8, gate_k=16, k=-1)
            (fc2): SingularMoELinear(in_features=3072, out_features=768, num_experts=8, top_k=8, gate_k=16, k=-1)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
)
```
