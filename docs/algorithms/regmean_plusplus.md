# RegMean++

[![arxiv](https://img.shields.io/badge/arXiv-2508.03121-b31b1b.svg)](https://www.arxiv.org/abs/2508.03121)
[![github](https://img.shields.io/badge/GitHub-Code-181717.svg)](https://github.com/nthehai01/RegMean-plusplus)

## Revisiting the RegMean Algorithm
**Regression Mean (RegMean)**[^1], an approach that formulates model merging as a linear regression problem, aims to find the optimal weights for each linear layer in the merge model by minimizing the discrepancy in predictions between the merge and candidate models. At a transformer layer $l$, to obtain the merge weights for a linear layer $W^{(l)}_{M}$ , RegMean provides a precise closed-form solution for merging those from $K$ candidate models as follows:

$$W^{(l)}_{M} = \left[\sum_{i=1}^{K}  (X^{(l)}_i)^{\top} X^{(l)}_i\right]^{-1} \sum_{i=1}^{K} (X^{(l)}_i)^{\top} X^{(l)}_i W^{(l)}_i.$$

## Problem of RegMean and How RegMean++ Addresses It
RegMean merges each linear layer independently, overlooking how the features and information in the earlier layers propagate through the layers and influence the final prediction in the merge model. To address this, **RegMean++**[^2] is proposed to explicitly incorporate both *intra- and cross-layer dependencies between merge models' layers* into RegMean's objective.


<figure markdown="span">
  ![alt text](images/regmean_vs_regmean_plusplus.png){ width="750" }
  <figcaption>
  <em><b>Comparison between RegMean and RegMean++ for model merging.</b> RegMean++ leverages representations from the merge model for merging, enabling accurate alignment with its behavior.</em>
  </figcaption>
</figure>

The key difference between RegMean++ and RegMean lies in how input feature $X^{(l,j)}_i$ for the $j$-th linear layer is obtained: *For input features that are **activations** (cushion representations between transformer layers), RegMean++ computes $X^{(l,j)}_i$ based on the activations produced by the **previous merge layer** $f_{M}^{(l-1)}$ in the merge model, that is, $X^{(l)}_i = f_{M}^{(l-1)}(X^{(l-1)}_{i})$ while RegMean relies on the activations produced by the **previous candidate layer** $f_{i}^{(l-1)}$ in the candidate model, that is, $X^{(l)}_i = f_{i}^{(l-1)}(X^{(l-1)}_{i})$.*


## Examples

### CLI Usage

The following command lines can be used to run and evaluate the RegMean++ algorithm on eight image classification tasks:

* For CLIP-ViT-B/32 models:
```bash
fusion_bench \
    method=regmean_plusplus/clip_regmean_plusplus \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32
```

* For CLIP-ViT-B/16 models:
```bash
fusion_bench \
    method=regmean_plusplus/clip_regmean_plusplus \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16
```

* For CLIP-ViT-L/14 models:
```bash
fusion_bench \
    method=regmean_plusplus/clip_regmean_plusplus \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
```

## Citation

```bibtex
@article{nguyen2025regmean++,
  title={RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging},
  author={Nguyen, The-Hai and Huu-Tien, Dang and Suzuki, Takeshi and Nguyen, Le-Minh},
  journal={arXiv preprint arXiv:2508.03121},
  year={2025}
}
```

## Implementation Details

- [RegMeanAlgorithmPlusPlus][fusion_bench.method.regmean_plusplus.RegMeanAlgorithmPlusPlus]
- [RegMeanAlgorithmForCLIPPlusPlus][fusion_bench.method.regmean_plusplus.RegMeanAlgorithmForCLIPPlusPlus]

[^1]: Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. "Dataless Knowledge Fusion by Merging Weights of Language Models." The Eleventh International Conference on Learning Representations.

[^2]: The-Hai Nguyen, Huu-Tien Dang, Takeshi Suzuki, and Le-Minh Nguyen. "RegMean++: Enhancing Effectiveness and Generalization of Regression Mean for Model Merging". arXiv preprint arXiv:2508.03121 (2025).
