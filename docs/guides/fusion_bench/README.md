# Fusion Module Documentation

- **method**: Implements different methods and algorithms for model training, evaluation, and other tasks. This includes base algorithms, ensemble methods, model recombination techniques, and more. [:octicons-arrow-right-24: Read More](../../algorithms/README.md)
- **modelpool**: Manages different model pools. This module includes classes and functions for handling various models, including sequence-to-sequence language models, CLIP models, GPT-2 models, and models specific to the NYU Depth V2 dataset. [:octicons-arrow-right-24: Read More](../../modelpool/README.md)
- **taskpool**: Manages different task pools. This module includes classes and functions for handling various tasks, such as image classification with CLIP, text generation with FLAN-T5, and tasks specific to the NYU Depth V2 dataset. [:octicons-arrow-right-24: Read More](../../taskpool/README.md)
- **models**: Contains model definitions and utilities. This module includes implementations of different models, parameter management, input/output handling, and utility functions for model operations.
- **tasks**: Defines various tasks. This module includes implementations of different tasks, such as classification, text generation, and specific tasks for models like CLIP and FLAN-T5.
- **dataset**: Handles various datasets used in the project. This module includes dataset loaders and preprocessors for different types of data such as images, text, and specific datasets like NYU Depth V2.
- **metrics**: Defines metrics for evaluating models. This module includes specific metrics for different tasks and datasets, such as NYU Depth V2 and text-to-image generation.
- **optim**: Defines optimization algorithms. This module includes custom optimization algorithms used for training models.
- **mixins**: Provides mixin classes for additional functionalities. These mixins can be used to extend the capabilities of other classes, such as integrating with Lightning Fabric, providing live updates with the Rich library, and simple profiling.
    - `LightningFabricMixin`: A mixin class for integrating Lightning Fabric into a project. [:octicons-arrow-right-24: Read More](mixins/lightning_fabric.md)
    - `SimpleProfilerMixin`: Adding profiling capabilities to your Python classes. [:octicons-arrow-right-24: Read More](mixins/simple_profiler.md)
- **constants**: Contains constant values and configurations used throughout the project.
- **scripts**: Contains scripts for various tasks. This includes command-line interface scripts, training scripts, and other utility scripts for managing and running different tasks.
- **utils**: Provides utility functions and classes. This module includes general utilities for data handling, device management, logging, timing, and other common operations needed throughout the project.
