import logging
import os
from typing import List, Literal, Optional, Union, TYPE_CHECKING

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
