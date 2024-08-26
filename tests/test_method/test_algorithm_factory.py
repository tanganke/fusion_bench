import unittest
from unittest.mock import MagicMock

from omegaconf import DictConfig

from fusion_bench.constants.stats import AVAILABLE_ALGORITHMS
from fusion_bench.method import load_algorithm_from_config
from fusion_bench.method.base_algorithm import ModelFusionAlgorithm


class TestAlgorithmFactory(unittest.TestCase):
    def setUp(self):
        # Mock configuration for testing
        self.mock_config = DictConfig({"name": ""})

    def test_load_all_algorithms(self):
        for algorithm_name in AVAILABLE_ALGORITHMS:
            with self.subTest(algorithm=algorithm_name):
                self.mock_config.name = algorithm_name
                try:
                    algorithm_instance = load_algorithm_from_config(self.mock_config)
                    self.assertIsInstance(algorithm_instance, ModelFusionAlgorithm)
                except Exception as e:
                    self.fail(
                        f"Loading algorithm '{algorithm_name}' raised an exception: {e}"
                    )


if __name__ == "__main__":
    unittest.main()
