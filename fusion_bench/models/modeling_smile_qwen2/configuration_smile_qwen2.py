from transformers import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class SmileQwen2Config(Qwen2Config):
    model_type = "smile_qwen2"

    def __init__(
        self,
        num_experts_per_tok: int = 1,
        rank_of_router: int = None,
        rank_of_expert: int = None,
        num_local_experts: int = None,
        **kwargs,
    ):
        self.num_experts_per_tok = num_experts_per_tok
        self.rank_of_router = rank_of_router
        self.rank_of_expert = rank_of_expert
        self.num_local_experts = num_local_experts

        super().__init__(**kwargs)
