# WUDI-Merging

[![arXiv](https://img.shields.io/badge/arXiv-2503.08099-b31b1b.svg)](http://arxiv.org/abs/2503.08099)

**WUDI-Merging** (Whoever started the interference shoUld enD It) is a novel data-free model merging method that minimizes interference between task vectors without requiring additional data or rescaling coefficients. The method is based on the theoretical insight that task vectors of linear layers constitute an approximate linear subspace for their corresponding inputs.

## Examples

### CLI Usage

Merging eight CLIP models trained on different classification tasks:

```bash
fusion_bench \
    method=wudi/wudi \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### API Usage

```python
from fusion_bench.method import WUDIMerging
from fusion_bench.modelpool import CLIPVisionModelPool

# Initialize the method
method = WUDIMerging(iter_num=400, exclude_keys=None)

# Load model pool
modelpool = CLIPVisionModelPool(
    models={
        "model1": "path/to/model1",
        "model2": "path/to/model2",
        # ... more models
    }
)

# Run merging
merged_model = method.run(modelpool)
```

## Implementation Details

- [fusion_bench.method.WUDIMerging][]
- Implementation: Added in pull request [#144](https://github.com/tanganke/fusion_bench/pull/144)
