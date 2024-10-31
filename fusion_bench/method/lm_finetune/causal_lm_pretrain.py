from fusion_bench import BaseAlgorithm
from fusion_bench.modelpool import CausalLMPool


class CausalLMPretrain(BaseAlgorithm):
    def run(self, modelpool: CausalLMPool):
        tokenizer = modelpool.load_tokenizer()
