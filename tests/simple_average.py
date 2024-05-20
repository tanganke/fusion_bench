from fusion_bench.method import SimpleAverageAlgorithm

# Instantiate the SimpleAverageAlgorithm
# This algorithm will be used to merge multiple models by averaging their parameters.
algorithm = SimpleAverageAlgorithm()

# Assume we have a list of PyTorch models (nn.Module instances) that we want to merge.
# The models should all have the same architecture.
from torch import nn

models = [nn.Linear(10, 10) for _ in range(10)]

# Run the algorithm on the models.
# This will return a new model that is the result of averaging the parameters of the input models.
merged_model = algorithm.run(models)
