import logging
import os
from typing import Optional

from datasets import load_dataset, load_from_disk

from fusion_bench.utils import timeit_context

from .glue_preprocessors import glue_processors
from .glue_prompt_templates import glue_prompt_templates

log = logging.getLogger(__name__)


def _load_glue_dataset(name, tokenizer):
    dataset = load_dataset("glue", name)
    preprocessor = glue_processors[name](
        template=glue_prompt_templates[name],
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        },
    )
    dataset = dataset.map(
        preprocessor,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    return dataset


def load_glue_dataset(name, tokenizer, cache_dir: Optional[str]):
    with timeit_context(f"Loading {name} dataset"):
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_path = os.path.join(
                cache_dir, "flan-t5", f"_load_{name}_dataset_cached"
            )
            if os.path.exists(cache_path):
                return load_from_disk(cache_path)
            else:
                dataset = _load_glue_dataset(name, tokenizer)
                log.info(f"Saving {name} dataset to {cache_path}")
                dataset.save_to_disk(cache_path)
                return dataset
        else:
            return _load_glue_dataset(name, tokenizer)
