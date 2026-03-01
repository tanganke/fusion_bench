from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


class SmileQwen3VLVisionConfig(Qwen3VLVisionConfig):
    model_type = "smile_qwen3_vl"
    base_config_key = "vision_config"

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


class SmileQwen3VLTextConfig(Qwen3VLTextConfig):
    model_type = "smile_qwen3_vl_text"
    base_config_key = "text_config"

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


class SmileQwen3VLConfig(Qwen3VLConfig):
    model_type = "smile_qwen3_vl"
    sub_configs = {
        "vision_config": SmileQwen3VLVisionConfig,
        "text_config": SmileQwen3VLTextConfig,
    }

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


__all__ = [
    "SmileQwen3VLConfig",
    "SmileQwen3VLVisionConfig",
    "SmileQwen3VLTextConfig",
]
