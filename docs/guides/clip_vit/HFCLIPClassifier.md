# Image Classification with CLIP Models using `HFCLIPClassifier`

## Introduction

The `HFCLIPClassifier` class provides a wrapper around the CLIP (Contrastive Language-Image Pre-training) model for image classification tasks. It supports zero-shot learning and can be fine-tuned for specific classification tasks.

## Basic Steps

### Importing Required Modules

First, we need to import the necessary modules for our CLIP-based image classification task:

```python
import torch
from transformers import CLIPModel, CLIPProcessor
from fusion_bench.models.hf_clip import HFCLIPClassifier
from torch.utils.data import DataLoader
```

### Loading CLIP Model and Processor

```python
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

### Initializing HFCLIPClassifier

```python
classifier = HFCLIPClassifier(clip_model, processor)
```

### Setting Up the Classification Task

After initializing the classifier, we need to set up the classification task by defining class names and optionally, custom text templates. 
The text encoder of CLIP model is used to encode the class names into text embeddings, which are then used to compute the logits for each class.


```python
class_names = ["cat", "dog", "bird", "fish", "horse"]
classifier.set_classification_task(class_names)
```

By default, `set_classification_task` uses the following templates:

```python
default_templates = [
    lambda c: f"a photo of a {c}",
]
```

You can also use custom templates:

```python
custom_templates = [
    lambda c: f"a photo of a {c}",
    lambda c: f"an image containing a {c}",
]
classifier.set_classification_task(class_names, templates=custom_templates)
```

Below is the code for `set_classification_task` and `forward` method of `HFCLIPClassifier`:

::: fusion_bench.models.hf_clip.HFCLIPClassifier
    options:
        members: 
        - set_classification_task
        - forward

## Preparing Your Dataset

Create a custom dataset class that loads and preprocesses your images:

```python
from torchvision import transforms
from PIL import Image

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: List[str], labels: List[int]):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]

# Create DataLoader
dataset = SimpleDataset(image_paths, labels)  # Replace with your data
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

You can also use `fusion_bench.dataset.clip_dataset.CLIPDataset` or `fusion_bench.dataset.image_dataset.TransformedImageDataset` to prepare your dataset. Here is examples of using `fusion_bench.dataset.clip_dataset.CLIPDataset` and `fusion_bench.dataset.image_dataset.TransformedImageDataset` to prepare your dataset:

```python
from fusion_bench.dataset.clip_dataset import CLIPDataset

dataset = CLIPDataset(dataset, processor)
```

```python
from fusion_bench.dataset.image_dataset import TransformedImageDataset

dataset = TransformedImageDataset(dataset, transform)
```

Where `dataset` is your original dataset and `transform` is the transform you want to apply to the images.
Below is the reference code for these two classes:

::: fusion_bench.dataset.clip_dataset.CLIPDataset
::: fusion_bench.dataset.image_dataset.TransformedImageDataset

## Inference

Perform inference on your dataset:

```python
classifier.eval()
with torch.no_grad():
    for images, labels in dataloader:
        logits = classifier(images)
        predictions = torch.argmax(logits, dim=1)
        # Process predictions as needed
```

## Fine-tuning (Optional)

If you want to fine-tune the model:

```python
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

classifier.train()
for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        logits = classifier(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
```

## Advanced Usage

### Custom Templates

You can provide custom templates when setting up the classification task:

```python
custom_templates = [
    lambda c: f"a photo of a {c}",
    lambda c: f"an image containing a {c}",
]
classifier.set_classification_task(class_names, templates=custom_templates)
```

### Accessing Model Components

You can access the text and vision models directly:

```python
text_model = classifier.text_model
vision_model = classifier.vision_model
```

### Working with Zero-shot Weights

After setting the classification task, you can access the zero-shot weights:

```python
zeroshot_weights = classifier.zeroshot_weights
```

These weights represent the text embeddings for each class and can be used for further analysis or custom processing.

Remember to adjust the code according to your specific dataset and requirements. This documentation provides a comprehensive guide for using the `HFCLIPClassifier` for image classification tasks with CLIP models.
