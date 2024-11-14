"""
This module provides the `DataAdaptiveWeightEnsemblingModel` class for data-adaptive weight ensembling.

The DataAdaptiveWeightEnsemblingModel class is designed to perform data-adaptive weight ensembling
for model fusion. It supports both task-wise and layer-wise merging modes and allows for the use
of different feature extractors and processors.

Classes:
    DataAdaptiveWeightEnsemblingModel: A class for data-adaptive weight ensembling.
"""

# flake8: noqa F401
from .dawe_model import DataAdaptiveWeightEnsemblingModel
