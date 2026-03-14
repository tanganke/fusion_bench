from fusion_bench.method.task_arithmetic.task_arithmetic import TaskArithmeticAlgorithm

from .svc import SingularValueCalibration
from typing import Optional, Union
import torch
from fusion_bench import auto_register_config, BaseModelPool


@auto_register_config
class SingularValueCalibrationArithmeticTask(SingularValueCalibration):
    """
    Combines Task Arithmetic merging with Singular Value Calibration (SVC).

    This class first merges the models in the pool using Task Arithmetic and
    then applies SVC to reduce spectral over-accumulation in the merged model.
    """

    def __init__(
        self,
        scaling_factor: float,
        alpha: float,
        accelerator: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """
        Initializes the Singular Value Calibration method for arithmetic tasks.

        Args:
            scaling_factor (float): Scaling factor for the task arithmetic step.
            alpha (float): Calibration strength hyperparameter that controls how much to scale down
                the merged responses along shared spectral subspaces. Higher values lead to more aggressive
                calibration, while lower values retain more of the original merged responses. Default is 1.0.
            accelerator: Optional device to perform computations on (e.g., ``'cuda'`` or ``'cpu'``).
                If ``None``, the device is selected automatically (CUDA > MPS > CPU).
        """
        super().__init__(alpha=alpha, accelerator=accelerator, **kwargs)
        self.scaling_factor = scaling_factor

    def run(self, modelpool: BaseModelPool):
        """
        Runs the Singular Value Calibration method on the given model pool, applying task arithmetic.

        Args:
            modelpool (BaseModelPool): The pool of models to calibrate.

        Returns:
            nn.Module: The calibrated merged model with task arithmetic applied.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        task_arithmetic_algorithm = TaskArithmeticAlgorithm(
            scaling_factor=self.scaling_factor, inplace=False
        )
        merged_model = task_arithmetic_algorithm.run(modelpool)
        modelpool.add_model("_merged_", merged_model)

        calibrated_model = super().run(modelpool)
        return calibrated_model
