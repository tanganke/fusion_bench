
Here we provide a basic example to demonstrate the usage of FusionBench.
We choose the [simple average algorithm](algorithms/simple_averaging.md) as the fusion algorithm, and 4 [fine-tuned CLIP-ViT-B/32 models](modelpool/clip_vit.md) to be merged.
We are going to evaluate the merged model on 4 tasks with data corrupted by Gaussian noise to evaluate the robustness of the merged model.

We provide an command line interface `fusion_bench` to run the example.
The instruction to run the example is as follows:

```{.bash .annotate}
fusion_bench \
    # (1)!
    --config-name clip-vit-base-patch32_robustness_corrupted \
    corruption=gaussian_noise \
    # (2)!
    method=simple_averaging  \
    # (3)!
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted \
    # (4)!
    taskpool=clip-vit-base-patch32_robustness_corrupted
```

1. Here we specify the main configuration file to be used.
    The `corruption` option specifies the type of data corruption to be applied to the evaluation datasets. In this case, we use Gaussian noise.
    In FusionBench, we are currently provide 7 types of data corruptions for imaage classification tasks Standford Cars, EuroSAT, RESISC45 and GTSRB.
    The option `corrption` can be one of: `contrast`, `gaussian_noise`, `impulse_noise`, `jpeg_compression`, `motion_blur`, `pixelate`, `spatter`.
2. The `method` option specifies the fusion algorithm to be used. In this case, we use the simple averaging algorithm.
3. Here we specify the model pool to be used.
    The model pool is responsible for managing the loading, preprocessing, and saving of the models.
    By pass option `modelpool=CLIPVisionModelPool/clip-vit-base-patch32_robustness_corrupted`, the program instantiate a modelpool object that manages 4 task-specific CLIP-ViT-B/32 models that are fine-tuned on Stanford Cars, EuroSAT, RESISC45, and GTSRB datasets.
4. Here we specify the task pool to be used.
    The task pool is responsible for managing the evaluation datasets and metrics.
    By pass option `taskpool=clip-vit-base-patch32_robustness_corrupted`, the program instantiate a taskpool object that manages 4 tasks with data corrupted by Gaussian noise.

The configurations are stored in the `configs` directory, listed as follows:

=== "Method Configuration"

    The simple averaging algorithm is very straightforward. No additional hyperparameters are required. So the configuration file contains only the name of the algorithm to specify the Python class of the fusion algorithm.

    ```yaml title="config/method/simple_average.yaml"
    name: simple_average # (1)
    ```

    1. Name of the fusion algorithm. The `name` field specifies the class of the fusion algorithm.

=== "Model Pool Configuration"

    ```yaml title="config/modelpool/clip-vit-base-patch32_robustness_corrupted.yaml"
    type: huggingface_clip_vision # (1)
    models: # (2)
    - name: _pretrained_
        path: openai/clip-vit-base-patch32
    - name: stanford_cars
        path: tanganke/clip-vit-base-patch32_stanford-cars
    - name: eurosat
        path: tanganke/clip-vit-base-patch32_eurosat
    - name: resisc45
        path: tanganke/clip-vit-base-patch32_resisc45
    - name: gtsrb
        path: tanganke/clip-vit-base-patch32_gtsrb


    # `corrption` can be one of:
    # contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, spatter
    corruption: ${corruption}

    # Other configurations to meet other methods' requirements.
    # For example, test dataset for test-time adaptation training.
    # ...
    ```

    1. Type of the model pool. The `type` field specifies the class of the model pool.
    2. The `models` field specifies the models to be used for fusion. In this case, we use 4 task-specific CLIP-ViT-B/32 models that are fine-tuned on Stanford Cars, EuroSAT, RESISC45, and GTSRB datasets.

=== "Task Pool Configuration"

    ```yaml title="config/taskpool/clip-vit-base-patch32_robustness_corrupted.yaml"
    type: clip_vit_classification # (1)
    name: clip-vit-robustness_clean

    # corrption can be one of:
    # contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, spatter
    corruption: ${corruption}
    dataset_type: huggingface_image_classification
    tasks: # (2)
    - name: stanford_cars
        dataset:
        name: tanganke/stanford_cars
        split: ${taskpool.corruption}
    - name: eurosat
        dataset:
        name: tanganke/eurosat
        split: ${taskpool.corruption}
    - name: resisc45
        dataset:
        name: tanganke/resisc45
        split: ${taskpool.corruption}
    - name: gtsrb
        dataset:
        name: tanganke/gtsrb
        split: ${taskpool.corruption}

    clip_model: openai/clip-vit-base-patch32 # (3)
    batch_size: 128 # (4)
    num_workers: 16
    fast_dev_run: ${fast_dev_run}
    ```

    1. Type and name of the task pool. The `type` field specifies the class of the task pool, and the `name` field specifies the name of the task pool.
    2. The `tasks` field specifies the tasks to be evaluated. In this case, we evaluate the fused model on 4 tasks: Stanford Cars, EuroSAT, RESISC45, and GTSRB, with data corrupted by `${corruption}`.
    3. Base model used for intializing the classification head. Here, we need the text encoder of CLIP-ViT-B/32 to initialize the classification head.
    4. Batch size and number of workers used for data loading.