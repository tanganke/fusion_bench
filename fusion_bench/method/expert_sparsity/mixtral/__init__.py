R"""
```bash
fusion_bench \
    modelpool=CausalLMPool/mixtral-8x7b \
    ...
```

if use flash attention 2, pass the following to the command line:

```bash
+modelpool.models._pretrained_.attn_implementation=flash_attention_2
```
"""

from .dynamic_skipping import DynamicSkippingPruningForMixtral
from .layer_wise_pruning import LayerWisePruningForMixtral
from .progressive_pruning import ProgressivePruningForMixtral

__all__ = [
    "DynamicSkippingPruningForMixtral",
    "LayerWisePruningForMixtral",
    "ProgressivePruningForMixtral",
]
