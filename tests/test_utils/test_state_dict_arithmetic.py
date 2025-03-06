import timeit
import unittest

import torch
from torch import nn

from fusion_bench.utils import timeit_context
from fusion_bench.utils.state_dict_arithmetic import *


def create_test_model():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100),
    )


class TestStateDictArithmetic(unittest.TestCase):
    def setUp(self):
        self.model_1 = create_test_model()
        self.model_2 = create_test_model()
        self.model_3 = create_test_model()
        if torch.cuda.is_available():
            self.model_1.cuda()
            self.model_2.cuda()
            self.model_3.cuda()

    def test_state_dict_sum(self):
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        state_dict_3 = self.model_3.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_sum([state_dict_1, state_dict_2, state_dict_3]),
            number=10,
        )
        print(f"Time taken for state_dict_sum: {time_taken} seconds")

    def test_state_dict_avg(self):
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        state_dict_3 = self.model_3.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_avg([state_dict_1, state_dict_2, state_dict_3]),
            number=10,
        )
        print(f"Time taken for state_dict_avg: {time_taken} seconds")

    def test_state_dict_sub(self):
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_sub(state_dict_1, state_dict_2),
            number=10,
        )
        print(f"Time taken for state_dict_sub: {time_taken} seconds")

    def test_state_dict_add(self):
        state_dict_1 = self.model_1.state_dict()
        state_dict_2 = self.model_2.state_dict()
        time_taken = timeit.timeit(
            lambda: state_dict_add(state_dict_1, state_dict_2),
            number=10,
        )
        print(f"Time taken for state_dict_add: {time_taken} seconds")

    def test_state_dict_mul(self):
        state_dict_1 = self.model_1.state_dict()
        scalar = 2.0
        time_taken = timeit.timeit(
            lambda: state_dict_mul(state_dict_1, scalar),
            number=10,
        )
        print(f"Time taken for state_dict_mul: {time_taken} seconds")


if __name__ == "__main__":
    unittest.main()
