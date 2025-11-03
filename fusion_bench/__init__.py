# ███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗      ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
# ██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║      ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
# █████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║█████╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
# ██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║╚════╝██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
# ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║      ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
# ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝      ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
# flake8: noqa: F401
import sys
from typing import TYPE_CHECKING

from fusion_bench.utils.lazy_imports import LazyImporter

from . import constants, metrics, optim, tasks
from .constants import RuntimeConstants
from .method import _available_algorithms

_extra_objects = {
    "RuntimeConstants": RuntimeConstants,
    "constants": constants,
    "metrics": metrics,
    "optim": optim,
    "tasks": tasks,
}
_import_structure = {
    "dataset": ["CLIPDataset"],
    "method": _available_algorithms,
    "mixins": [
        "CLIPClassificationMixin",
        "FabricTrainingMixin",
        "HydraConfigMixin",
        "LightningFabricMixin",
        "OpenCLIPClassificationMixin",
        "PyinstrumentProfilerMixin",
        "SimpleProfilerMixin",
        "YAMLSerializationMixin",
        "auto_register_config",
    ],
    "modelpool": [
        "AutoModelPool",
        "BaseModelPool",
        "CausalLMBackbonePool",
        "CausalLMPool",
        "CLIPVisionModelPool",
        "ConvNextForImageClassificationPool",
        "Dinov2ForImageClassificationPool",
        "GPT2ForSequenceClassificationPool",
        "HuggingFaceGPT2ClassificationPool",
        "NYUv2ModelPool",
        "OpenCLIPVisionModelPool",
        "PeftModelForSeq2SeqLMPool",
        "ResNetForImageClassificationPool",
        "Seq2SeqLMPool",
        "SequenceClassificationModelPool",
    ],
    "models": [
        "create_default_model_card",
        "load_model_card_template",
        "save_pretrained_with_remote_code",
        "separate_load",
        "separate_save",
    ],
    "programs": ["BaseHydraProgram", "FabricModelFusionProgram"],
    "taskpool": [
        "BaseTaskPool",
        "CLIPVisionModelTaskPool",
        "DummyTaskPool",
        "GPT2TextClassificationTaskPool",
        "LMEvalHarnessTaskPool",
        "OpenCLIPVisionModelTaskPool",
        "NYUv2TaskPool",
    ],
    "utils": [
        "ArithmeticStateDict",
        "BoolStateDictType",
        "LazyStateDict",
        "StateDictType",
        "TorchModelType",
        "cache_with_joblib",
        "get_rankzero_logger",
        "import_object",
        "instantiate",
        "parse_dtype",
        "print_parameters",
        "seed_everything_by_time",
        "set_default_cache_dir",
        "set_print_function_call",
        "set_print_function_call_permeanent",
        "timeit_context",
    ],
}

if TYPE_CHECKING:
    from .dataset import CLIPDataset
    from .method import BaseAlgorithm, BaseModelFusionAlgorithm
    from .mixins import (
        CLIPClassificationMixin,
        FabricTrainingMixin,
        HydraConfigMixin,
        LightningFabricMixin,
        OpenCLIPClassificationMixin,
        PyinstrumentProfilerMixin,
        SimpleProfilerMixin,
        YAMLSerializationMixin,
        auto_register_config,
    )
    from .modelpool import (
        AutoModelPool,
        BaseModelPool,
        CausalLMBackbonePool,
        CausalLMPool,
        CLIPVisionModelPool,
        ConvNextForImageClassificationPool,
        Dinov2ForImageClassificationPool,
        GPT2ForSequenceClassificationPool,
        HuggingFaceGPT2ClassificationPool,
        NYUv2ModelPool,
        OpenCLIPVisionModelPool,
        PeftModelForSeq2SeqLMPool,
        ResNetForImageClassificationPool,
        Seq2SeqLMPool,
        SequenceClassificationModelPool,
    )
    from .models import (
        create_default_model_card,
        load_model_card_template,
        save_pretrained_with_remote_code,
        separate_load,
        separate_save,
    )
    from .programs import BaseHydraProgram, FabricModelFusionProgram
    from .taskpool import (
        BaseTaskPool,
        CLIPVisionModelTaskPool,
        DummyTaskPool,
        GPT2TextClassificationTaskPool,
        LMEvalHarnessTaskPool,
        NYUv2TaskPool,
        OpenCLIPVisionModelTaskPool,
    )
    from .utils import (
        ArithmeticStateDict,
        BoolStateDictType,
        LazyStateDict,
        StateDictType,
        TorchModelType,
        cache_with_joblib,
        get_rankzero_logger,
        import_object,
        instantiate,
        parse_dtype,
        print_parameters,
        seed_everything_by_time,
        set_default_cache_dir,
        set_print_function_call,
        set_print_function_call_permeanent,
        timeit_context,
    )
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects=_extra_objects,
    )
