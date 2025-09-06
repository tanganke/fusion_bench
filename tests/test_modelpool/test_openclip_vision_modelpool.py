from fusion_bench.utils.packages import is_open_clip_available

if is_open_clip_available():
    import logging
    import os
    import unittest

    import torch
    from hydra import initialize
    from omegaconf import OmegaConf
    from torch import nn

    from fusion_bench.modelpool import OpenCLIPVisionModelPool
    from fusion_bench.utils import instantiate

    log = logging.getLogger(__name__)

    class TestOpenCLIPVisionModelPool(unittest.TestCase):
        def setUp(self):
            config_path = "../../config"
            if not os.path.exists(os.path.join(os.path.dirname(__file__), config_path)):
                raise RuntimeError(f"Config path {config_path} does not exist.")
            else:
                log.info(f"set config path to {config_path}")

            with initialize(version_base=None, config_path=config_path):
                vit_b_32_ta8_modelpool: OpenCLIPVisionModelPool = (
                    OpenCLIPVisionModelPool.from_config(
                        "modelpool/OpenCLIPVisionModelPool/ViT-B-32_TA8.yaml",
                    )
                )
                vit_b_16_ta8_modelpool: OpenCLIPVisionModelPool = (
                    OpenCLIPVisionModelPool.from_config(
                        "modelpool/OpenCLIPVisionModelPool/ViT-B-16_TA8.yaml",
                    )
                )
                vit_l_14_ta8_modelpool: OpenCLIPVisionModelPool = (
                    OpenCLIPVisionModelPool.from_config(
                        "modelpool/OpenCLIPVisionModelPool/ViT-L-14_TA8.yaml",
                    )
                )
            self.vit_b_32_ta8_modelpool = vit_b_32_ta8_modelpool
            self.vit_b_16_ta8_modelpool = vit_b_16_ta8_modelpool
            self.vit_l_14_ta8_modelpool = vit_l_14_ta8_modelpool

        def _test_load_model(self, modelpool: OpenCLIPVisionModelPool):
            # try to load all models
            model = modelpool.load_pretrained_model()
            self.assertIsNotNone(model)

            for model_name in modelpool.model_names:
                model = modelpool.load_model(model_name)
                self.assertIsNotNone(model)

        def test_load_model(self):
            self._test_load_model(self.vit_b_32_ta8_modelpool)
            self._test_load_model(self.vit_b_16_ta8_modelpool)
            self._test_load_model(self.vit_l_14_ta8_modelpool)

        def _test_load_classification_head(self, modelpool: OpenCLIPVisionModelPool):
            for model_name in modelpool.model_names:
                head = modelpool.load_classification_head(model_name)
                self.assertIsNotNone(head)

        def test_load_classification_head(self):
            self._test_load_classification_head(self.vit_b_32_ta8_modelpool)
            self._test_load_classification_head(self.vit_b_16_ta8_modelpool)
            self._test_load_classification_head(self.vit_l_14_ta8_modelpool)

        def test_train_processor(self):
            modelpool = self.vit_b_32_ta8_modelpool
            processor = modelpool.train_processor
            self.assertIsNotNone(processor)

        def test_test_processor(self):
            modelpool = self.vit_b_32_ta8_modelpool
            processor = modelpool.test_processor
            self.assertIsNotNone(processor)

    if __name__ == "__main__":
        unittest.main()


else:
    import unittest

    class TestOpenCLIPVisionModelPool(unittest.TestCase):
        def test_import(self):
            import open_clip

    if __name__ == "__main__":
        unittest.main()
