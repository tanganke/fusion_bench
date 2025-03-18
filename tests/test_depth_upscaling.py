import unittest

from omegaconf import DictConfig
from torch import nn

from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method.depth_upscaling import DepthUpscalingAlgorithm


class TestDepthUpscalingAlgorithm(unittest.TestCase):
    def setUp(self):
        self.method_config = {"name": "depth_upscaling", "layer_indices": [0, 1, 1, 0]}
        self.algorithm = DepthUpscalingAlgorithm(DictConfig(self.method_config))

        # Assume we have a list of PyTorch models (nn.ModuleList instances) that we want to upscale.
        self.model = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])

    def test_run_valid_modelpool(self):
        upscaled_model = self.algorithm.run(self.model)
        self.assertIsNotNone(upscaled_model)
        self.assertIsInstance(upscaled_model, nn.ModuleList)
        self.assertEqual(len(upscaled_model), 4)

    def test_run_multiple_models(self):
        modelpool = to_modelpool({"model1": self.model, "model2": self.model})
        with self.assertRaises(AssertionError):
            self.algorithm.run(modelpool)

    def test_run_non_modulelist_model(self):
        modelpool = to_modelpool({"model1": nn.Linear(10, 10)})
        with self.assertRaises(AssertionError):
            self.algorithm.run(modelpool)

    def test_run_solar_model(self):
        from omegaconf import DictConfig
        from torch import nn
        from transformers import AutoModelForCausalLM, MistralConfig, MistralForCausalLM

        from fusion_bench.method.depth_upscaling import DepthUpscalingAlgorithm

        model_config = MistralConfig(
            # https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json
            **{
                "architectures": ["MistralForCausalLM"],
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "initializer_range": 0.02,
                "intermediate_size": 14336,
                "max_position_embeddings": 32768,
                "model_type": "mistral",
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-05,
                "rope_theta": 10000.0,
                "sliding_window": 4096,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.34.0.dev0",
                "use_cache": True,
                "vocab_size": 32000,
            }
        )
        model: MistralForCausalLM = AutoModelForCausalLM.from_config(model_config)

        method_config = {
            "name": "depth_upscaling",
            "layer_indices": ["range(0,24)", "range(8,32)"],
        }
        algorithm = DepthUpscalingAlgorithm(DictConfig(method_config))
        upscaled_model = algorithm.run(model.model.layers)

        # substitute the model with the upscaled model
        model.model.layers = upscaled_model


if __name__ == "__main__":
    unittest.main()
