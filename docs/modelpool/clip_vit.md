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

### Model Pool Configuration

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

### Basic Examples

Here are some basic examples of using the CLIP-ViT models for open vocabulary image classification with different fusion methods, using the [`fusion_bench`](../cli/fusion_bench.md) command line interface.

#### Simple Averaging

merge CLIP-ViT-B/32 models using simple average and evaluate on the eight tasks

```bash
fusion_bench method=simple_average \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8 

# results
{
    "svhn": {"accuracy": 0.6451674699783325, "loss": 1.128771424293518},
    "stanford_cars": {"accuracy": 0.625668466091156, "loss": 1.135254979133606},
    "resisc45": {"accuracy": 0.7079365253448486, "loss": 0.9697789549827576},
    "eurosat": {"accuracy": 0.7685185074806213, "loss": 0.6301173567771912},
    "gtsrb": {"accuracy": 0.5494061708450317, "loss": 1.492265224456787},
    "mnist": {"accuracy": 0.8626000285148621, "loss": 0.5933865308761597},
    "dtd": {"accuracy": 0.5090425610542297, "loss": 1.79731023311615},
    "sun397": {"accuracy": 0.6543576717376709, "loss": 1.1993952989578247},
}
```

merge CLIP-ViT-L/14 models using simple average and evaluate on the eight tasks

```bash
fusion_bench method=simple_average \
  modelpool=clip-vit-large-patch14_TA8 \
  taskpool=clip-vit-classification_TA8 taskpool.clip_model=openai/clip-vit-large-patch14 # because when evaluate the model, we need text encoder, so we need to specify the clip model
```

#### Task Arithmetic

merge CLIP-ViT-B/32 models using task arithmetic and evaluate on the eight tasks

```bash
fusion_bench method=task_arithmetic method.scaling_factor=0.3\
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8

# results
{
    "svhn": {"accuracy": 0.77927166223526, "loss": 0.7050645351409912},
    "stanford_cars": {"accuracy": 0.5565228462219238, "loss": 1.4873239994049072},
    "resisc45": {"accuracy": 0.6487301588058472, "loss": 1.3709946870803833},
    "eurosat": {"accuracy": 0.7674074172973633, "loss": 0.6550557017326355},
    "gtsrb": {"accuracy": 0.6850356459617615, "loss": 1.2349143028259277},
    "mnist": {"accuracy": 0.9606999754905701, "loss": 0.1570172756910324},
    "dtd": {"accuracy": 0.471808522939682, "loss": 2.1495635509490967},
    "sun397": {"accuracy": 0.571083128452301, "loss": 1.7016042470932007},
}
```

```bash
# or use a for loop to try different scaling factors 
# and save the results to different files
for scaling_factor in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  fusion_bench \
    method=task_arithmetic method.scaling_factor=$scaling_factor\
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    save_report=outputs/clip-vit-base-patch32_TA8_task_arithmetic_scaling_factor_${scaling_factor}.json
done
```

#### Ties-Merging

merge CLIP-ViT-B/32 models using Ties-Merging and evaluate on the eight tasks

```bash
fusion_bench method=ties_merging method.scaling_factor=0.3 method.threshold=20 \
  modelpool=clip-vit-base-patch32_TA8 \
  taskpool=clip-vit-classification_TA8
```

```bash
# or use a for loop to try different scaling factors
# and save the results to different files
for scaling_factor in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
  fusion_bench \
    method=ties_merging method.scaling_factor=$scaling_factor method.threshold=20 \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    save_report=outputs/clip-vit-base-patch32_TA8_ties_merging_scaling_factor_${scaling_factor}.json
done
```

### Experimental Results

We provide the experimental results of the CLIP-ViT models for open vocabulary image classification on the eight tasks in the following table.

| Model (CLIP-ViT-B/32)           | SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD  | Average |
| ------------------------------- | ------ | ---- | -------- | ------- | ---- | ----- | ----- | ---- | ------- |
| Pre-trained                     | 63.2   | 59.9 | 60.5     | 45.6    | 23.5 | 30.4  | 47.6  | 43.9 | 46.8    |
| Fine-tuned (STL)                | 75.0   | 78.2 | 95.2     | 99.1    | 97.1 | 98.8  | 99.6  | 79.7 | 90.3    |
| Simple Averaging                | 65.4   | 62.6 | 70.8     | 76.9    | 64.5 | 54.9  | 86.3  | 50.9 | 66.5    |
| Task Arithmetic ($\lambda=0.3$) |
| Ties-Merging ($\lambda=0.3$)    |

| Model (CLIP-ViT-L/14)           | SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD  | Average |
| ------------------------------- | ------ | ---- | -------- | ------- | ---- | ----- | ----- | ---- | ------- |
| Pre-trained                     | 68.3   | 77.7 | 71.0     | 61.5    | 58.8 | 43.8  | 76.0  | 55.5 | 64.1    |
| Fine-tuned (STL)                | 82.8   | 92.7 | 97.4     | 99.2    | 97.9 | 99.3  | 99.8  | 85.5 | 94.3    |
| Simple Averaging                |
| Task Arithmetic ($\lambda=0.3$) |
| Ties-Merging ($\lambda=0.3$)    |

