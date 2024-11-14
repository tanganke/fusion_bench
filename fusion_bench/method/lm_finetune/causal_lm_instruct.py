from fusion_bench import BaseAlgorithm
from fusion_bench.modelpool import CausalLMPool


class CausalLMInstructionFineTune(BaseAlgorithm):

    def run(self, modelpool: CausalLMPool):
        tokenizer = modelpool.load_tokenizer()
        model = modelpool.load_model()
        optimizer = modelpool.load_optimizer(model)
        scheduler = modelpool.load_scheduler(optimizer)
        dataloader = modelpool.load_dataloader(tokenizer)
        model = modelpool.train_model(model, dataloader, optimizer, scheduler)
        modelpool.save_model(model)
        return modelpool.evaluate_model(model)
