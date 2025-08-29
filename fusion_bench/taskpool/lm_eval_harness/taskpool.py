import logging
import os
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import lightning.fabric
import lm_eval
import lm_eval.models
from lm_eval.__main__ import check_argument_types, cli_evaluate, setup_parser
from omegaconf import DictConfig, ListConfig

from fusion_bench import BaseTaskPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils.strenum import _version

log = logging.getLogger(__name__)


class LMEvalHarnessTaskPool(BaseTaskPool, LightningFabricMixin):
    """A task pool implementation that interfaces with the LM Evaluation Harness framework.

    This class provides a wrapper around the LM Evaluation Harness (lm-eval) library,
    enabling evaluation of language models on various standardized benchmarks and tasks.
    It inherits from BaseTaskPool and LightningFabricMixin to provide distributed
    computing capabilities through PyTorch Lightning Fabric.

    The task pool supports evaluation on multiple tasks simultaneously and provides
    flexible configuration options for batch processing, output formatting, and
    logging. It automatically handles model setup and wrapping for distributed
    evaluation when using Lightning Fabric.

    Args:
        tasks: A single task name or list of task names to evaluate on.
            Examples: "hellaswag", ["arc_easy", "arc_challenge", "hellaswag"]
        apply_chat_template: Whether to apply chat template formatting to inputs.
            Useful for instruction-tuned or chat models.
        include_path: Path to additional task definitions or custom tasks.
        batch_size: Number of samples to process in each batch. Larger values
            may improve throughput but require more memory.
        metadata: Additional metadata to include in evaluation results.
        verbosity: Logging verbosity level for the evaluation process.
        output_path: Custom path for saving evaluation results. If None,
            results are saved to the default log directory.
        log_samples: Whether to log individual sample predictions and targets.
            Useful for debugging but increases output size significantly.
        _usage_: Internal usage tracking string.
        _version_: Internal version tracking string.
        **kwargs: Additional arguments passed to the LM Evaluation Harness.

    Example:
        ```python
        >>> taskpool = LMEvalHarnessTaskPool(
        ...     tasks=["arc_easy", "hellaswag"],
        ...     batch_size=8,
        ...     verbosity="INFO"
        ... )
        >>> results = taskpool.evaluate(model)
        ```
    """

    def __init__(
        self,
        tasks: Union[str, List[str]],
        apply_chat_template: bool = False,
        include_path: Optional[str] = None,
        batch_size: int = 1,
        metadata: Optional[DictConfig] = None,
        verbosity: Optional[
            Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
        ] = None,
        output_path: Optional[str] = None,
        log_samples: bool = False,
        _usage_: Optional[str] = None,
        _version_: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(_usage_=_usage_, _version_=_version_)
        self.tasks = tasks
        self.include_path = include_path
        self.batch_size = batch_size
        self.metadata = metadata
        self.apply_chat_template = apply_chat_template
        self.verbosity = verbosity
        self.kwargs = kwargs
        self.output_path = output_path
        self.log_samples = log_samples

    def evaluate(self, model, *command_line_args, **kwargs):
        """Evaluate a language model on the configured tasks using LM Evaluation Harness.

        This method wraps the model with the LM Evaluation Harness framework and
        executes evaluation on all configured tasks. It automatically handles
        command-line argument construction, model wrapping with Lightning Fabric
        for distributed evaluation, and result logging.

        The evaluation process includes:
        1. Building command-line arguments from instance configuration
        2. Setting up the LM Evaluation Harness argument parser
        3. Wrapping the model with Lightning Fabric if not already wrapped
        4. Creating an HFLM (Hugging Face Language Model) wrapper
        5. Executing the evaluation through the LM-Eval CLI interface

        Args:
            model: The language model to evaluate. Can be a Hugging Face model,
                PyTorch model, or any model compatible with the LM Evaluation Harness.
                The model will be automatically wrapped with Lightning Fabric for
                distributed evaluation if not already wrapped.
            *command_line_args: Additional positional command-line arguments
                (currently unused but preserved for interface compatibility).
            **kwargs: Additional keyword arguments that will be converted to
                command-line flags and passed to the LM Evaluation Harness.
                Keys will be prefixed with '--' and values converted to strings.

        Returns:
            None: Results are written to the configured output path and logged.

        Example:
            ```python
            >>> taskpool = LMEvalHarnessTaskPool(tasks=["arc_easy"])
            >>> taskpool.evaluate(model, limit=100, device="cuda")
            ```

        Note:
            The method leverages the LM Evaluation Harness's command-line interface
            internally, which provides standardized evaluation procedures and
            ensures compatibility with the broader evaluation ecosystem.
        """
        command_line_args = []
        if self.include_path is not None:
            command_line_args.extend(["--include_path", self.include_path])
        if isinstance(self.tasks, (list, ListConfig)):
            command_line_args.extend(["--tasks", ",".join(self.tasks)])
        elif isinstance(self.tasks, str):
            command_line_args.extend(["--tasks", self.tasks])
        if self.apply_chat_template:
            command_line_args.extend(
                ["--apply_chat_template", str(self.apply_chat_template)]
            )
        if self.batch_size is not None:
            command_line_args.extend(["--batch_size", str(self.batch_size)])
        if self.verbosity is not None:
            command_line_args.extend(["--verbosity", str(self.verbosity)])
        if self.metadata is not None:
            command_line_args.extend(["--metadata", str(self.metadata)])
        if self.output_path is None:
            command_line_args.extend(
                [
                    "--output_path",
                    os.path.join(self.log_dir, "lm_eval_results"),
                ]
            )
        else:
            command_line_args.extend(["--output_path", self.output_path])
        if self.log_samples:
            command_line_args.extend(["--log_samples"])
        for key, value in kwargs.items():
            command_line_args.extend([f"--{key}", str(value)])

        parser = setup_parser()
        check_argument_types(parser)
        args = parser.parse_args(args=command_line_args)
        log.info("LM-Eval Harness arguments:\n%s", args)

        if not lightning.fabric.is_wrapped(model):
            model = self.fabric.setup(model)
        args.model = lm_eval.models.huggingface.HFLM(pretrained=model)
        cli_evaluate(args)
