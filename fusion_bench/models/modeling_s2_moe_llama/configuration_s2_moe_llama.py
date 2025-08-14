from transformers.models.llama.configuration_llama import LlamaConfig


class S2MoELlamaConfig(LlamaConfig):
    model_type = "s2_moe_llama"

    def __init__(
        self,
        num_experts_per_tok: int = 1,
        num_local_experts: int = None,
        use_sparse_expert: bool = True,
        sparsity_ratio: float = 0.0,
        **kwargs,
    ):
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.use_sparse_expert = use_sparse_expert
        self.sparsity_ratio = sparsity_ratio

        super().__init__(**kwargs)
