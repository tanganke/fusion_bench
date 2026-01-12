from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

from .utils import EMRModulatedModel, EMRTaskModulator, emr_merge


@auto_register_config
class EMRMerging(BaseAlgorithm):
    """
    EMR Merging Algorithm.

    This algorithm merges multiple task-specific models into a unified model using
    the Elect, Mask & Rescale (EMR) strategy. It constructs a modulated model that
    can adapt to different tasks via task-specific modulators.
    """

    def load_pretrained_model_and_task_vectors(self, modelpool: BaseModelPool):
        pretrained_model = modelpool.load_pretrained_model()

        task_vectors = []
        for model_name in modelpool.model_names:
            finetuned_model = modelpool.load_model(model_name)
            task_vector = state_dict_sub(
                finetuned_model.state_dict(), pretrained_model.state_dict()
            )
            task_vectors.append(task_vector)

        return pretrained_model, task_vectors

    def run(self, modelpool: BaseModelPool) -> EMRModulatedModel:
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        pretrained_model, task_vectors = (
            modelpool.load_pretrained_model_and_task_vectors()
        )

        unified_vector, masks, rescalers = emr_merge(task_vectors)

        emr_model = EMRModulatedModel(
            backbone=pretrained_model, modulators={}, unified_task_vector=unified_vector
        )

        for model_idx, model_name in enumerate(modelpool.model_names):
            emr_model.add_modulator(
                task_name=model_name,
                modulator=EMRTaskModulator(
                    mask={n: m[model_idx] for n, m in masks.items()},
                    rescaler=rescalers[model_idx],
                ),
            )

        return emr_model
