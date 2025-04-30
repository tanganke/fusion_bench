from omegaconf import DictConfig

from fusion_bench import BaseAlgorithm
from fusion_bench.mixins import OpenCLIPClassificationMixin, SimpleProfilerMixin
from fusion_bench.modelpool import OpenCLIPVisionModelPool


class PWEMoEAlgorithmForOpenCLIP(
    BaseAlgorithm,
    SimpleProfilerMixin,
    OpenCLIPClassificationMixin,
):
    def __init__(
        self,
        *,
        upscale_mlp: bool,
        upscale_attn: bool,
        init_lambda: float,
        router_hidden_layers: int,
        lr: float,
        num_steps: int,
        save_interval: int,
        alpha: float,
        checkpoint_path: str,
        eval_grid: bool,
        eval_grid_n: int,
        eval_grid_m: int,
        dataloader_kwargs: DictConfig,
        run_train: bool,
        run_eval: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.upscale_mlp = upscale_mlp
        self.upscale_attn = upscale_attn
        self.init_lambda = init_lambda
        self.router_hidden_layers = router_hidden_layers
        self.lr = lr
        self.num_steps = num_steps
        self.save_interval = save_interval
        self.alpha = alpha
        self.checkpoint_path = checkpoint_path
        self.eval_grid = eval_grid
        self.eval_grid_n = eval_grid_n
        self.eval_grid_m = eval_grid_m
        self._dataloader_kwargs = dataloader_kwargs
        self.run_train = run_train
        self.run_eval = run_eval

    def run(self, modelpool: OpenCLIPVisionModelPool):
        self.modelpool = modelpool

        self.load_model()
        self.load_datasets()

        if self.run_train:
            self.train()
        if self.run_eval:
            self.evaluate()

    def load_model(self):
        pass

    def load_datasets(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass
