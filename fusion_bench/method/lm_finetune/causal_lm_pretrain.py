from fusion_bench import BaseModelFusionAlgorithm
from fusion_bench.modelpool import CausalLMPool


class CausalLMPretrain(BaseModelFusionAlgorithm):
    def run(self, modelpool: CausalLMPool):
        tokenizer = modelpool.load_tokenizer()
