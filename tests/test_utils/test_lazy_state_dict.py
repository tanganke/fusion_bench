import timeit
import unittest

import torch
from torch import nn
from transformers import CLIPModel, CLIPVisionModel

from fusion_bench.utils import LazyStateDict


class TestLazyStateDict(unittest.TestCase):
    def setUp(self):
        self.state_dict = LazyStateDict(
            "openai/clip-vit-base-patch32",
            meta_module_class=CLIPVisionModel,
        )
        self.clip_model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def test_consistency(self):
        # check if the keys are the same
        self.assertEqual(
            set(self.state_dict.keys()), set(self.clip_model.state_dict().keys())
        )
        # check if the values are the same
        for key in self.state_dict.keys():
            self.assertTrue(
                torch.allclose(self.state_dict[key], self.clip_model.state_dict()[key])
            )

    def test_values(self):
        num_values = 0
        for value in self.state_dict.values():
            self.assertIsInstance(value, torch.Tensor)
            num_values += 1
        self.assertEqual(num_values, len(self.clip_model.state_dict()))

    def test_keys(self):
        num_keys = 0
        for key in self.state_dict.keys():
            self.assertIsInstance(key, str)
            num_keys += 1
        self.assertEqual(num_keys, len(self.clip_model.state_dict()))
        self.assertEqual(
            set(self.state_dict.keys()), set(self.clip_model.state_dict().keys())
        )

    def test_items(self):
        for key, value in self.state_dict.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, torch.Tensor)

    def test_get_parameter(self):
        for key, value in self.clip_model.named_parameters():
            self.assertTrue(
                torch.allclose(
                    self.state_dict.get_parameter(key),
                    value,
                )
            )
