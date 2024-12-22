This folder contains the configuration for the CLIP-ViT models (managed by `fusion_bench.modelpool.CLIPVisionModelPool`).

## Expected Configuration

### Detailed Configuration


```yaml
${name_of_model}:
  _target_: ${function_to_load_model}
  ... # arguments to pass to the function
```

For example, to load the pre-trained CLIP-ViT-B/16 model, you can use the following configuration:

```yaml
_pretrained_: # `_pretrained_` is a special key in FusionBench that indicates the model is pre-trained
  _target_: transformers.CLIPVisionModel.from_pretrained
  pretrained_model_name_or_path: openai/clip-vit-base-patch16
```

In this case, calling `modelpool.load_model("_pretrained_")` will return a `transformers.CLIPVisionModel` instance, which is equivalent to call `transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")`.

The detailed configuration is more flexible and can be used when you need to pass additional arguments to the `from_pretrained` function or call custom functions to load and preprocess the model.

### Simplified Configuration

```yaml
${name_of_model}: ${pretrained_model_name_or_path}
```

This is a simplified configuration that is equivalent to the detailed configuration.

For example, to load the pre-trained CLIP-ViT-B/16 model, you can use the following configuration:

```yaml
_pretrained_: openai/clip-vit-base-patch16
```
