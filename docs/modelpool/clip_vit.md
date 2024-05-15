# CLIP-ViT Models for Open Vocabulary Image Classification

Here we provides a list of CLIP-ViT models that are trained for open vocabulary image classification. 

## The Eight Tasks

The most common eight tasks used in the research community are SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, and DTD.
These tasks cover a wide range of domains, including natural images, satellite images, and digit recognition.
You can download the datasets from [this HuggingFace Collection](https://huggingface.co/collections/tanganke/the-eight-image-classification-tasks-6644ce0376c0a469f6928507) or using the `datasets` library as follows:

```python
from datasets import load_dataset

# take `gtsrb` as an example
dataset = load_dataset("tanganke/gtsrb")

train_dataset = dataset["train"]
test_dataset = dataset["test"]
```

The authors of Task Arithmetic have fine-tuned the CLIP-ViT models from the *open_clip* library on these eight tasks and provide the models publicly on [Google Drive](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw?usp=share_link). 
However, these models rely on a specific version of the *open_clip* library. 

To make experiments more convenient and avoid dependency on a specific library version, we have re-trained these models and made them publicly available on the HuggingFace Model Hub.
We use the Adam Optimizer with a fixed learning rate of 1e-5 over 4000 training steps (batch_size=32).
Only the vision encoder is fine-tuned, while the text encoder remains fixed to preserve the open-vocabulary property of the model.

- [fine-tuned CLIP-ViT-B/32 models](https://huggingface.co/collections/tanganke/clip-vit-b-32-on-the-eight-image-classication-tasks-6644d0c476c0a469f693cf91)
- [fine-tuned CLIP-ViT-L/14 models](https://huggingface.co/collections/tanganke/clip-vit-l-14-on-the-eight-image-classification-tasks-6644d2b014331c746683de63)

To use these models, you can load them from the Transformers library as follows:

load vision backbone

```python
from transformers import CLIPVisionModel

# load the CLIP-ViT-B/32 model, take `gtsrb` as an example
vision_model = CLIPVisionModel.from_pretrained('tanganke/clip-vit-base-patch32_gtsrb')
```

substitute the vision encoder of clip

```python
from transformers import CLIPProcessor, CLIPModel

# load pre-trained CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# substitute the vision model with the fine-tuned one
clip_model.vision_model.load_state_dict(vision_model.vision_model.state_dict())
```

### Use Cases

To use these models from our FusionBench library, you can specify the modelpool configuration file as follows:

```yaml title="config/modelpool/clip-vit-base-patch32_TA8.yaml"
type: huggingface_clip_vision
models:
  - name: _pretrained_
    path: openai/clip-vit-base-patch32
  - name: sun397
    path: tanganke/clip-vit-base-patch32_sun397
  - name: stanford_cars
    path: tanganke/clip-vit-base-patch32_stanford-cars
  - name: resisc45
    path: tanganke/clip-vit-base-patch32_resisc45
  - name: eurosat
    path: tanganke/clip-vit-base-patch32_eurosat
  - name: svhn
    path: tanganke/clip-vit-base-patch32_svhn
  - name: gtsrb
    path: tanganke/clip-vit-base-patch32_gtsrb
  - name: mnist
    path: tanganke/clip-vit-base-patch32_mnist
  - name: dtd
    path: tanganke/clip-vit-base-patch32_dtd
```

The type of the modelpool is `huggingface_clip_vision`, corresponding to the modelpool class `HuggingFaceClipVisionPool`.

::: fusion_bench.modelpool.HuggingFaceClipVisionPool

