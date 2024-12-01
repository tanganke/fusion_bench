import logging

from transformers import AutoTokenizer

from .llama_3_Instruct import CHAT_TEMPLATE as LLAMA_3_INSTRUCT_CHAT_TEMPLATE

chat_template_mapping = {"llama_3_instruct": LLAMA_3_INSTRUCT_CHAT_TEMPLATE}

log = logging.getLogger(__name__)


def load_tokenizer_with_chat_template(
    pretrained_model_name_or_path: str,
    model_family: str,
    overwrite_chat_template: bool = True,
    **kwargs,
):
    """
    Load the tokenizer for Llama 3 model.

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.
        model_family (str): The model family.
        **kwargs: Additional keyword arguments passed to the tokenizer class.
    """
    assert (
        model_family in chat_template_mapping
    ), f"Model family {model_family} not found. Available model families: {chat_template_mapping.keys()}"

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        **kwargs,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template_mapping[model_family]
    else:
        if overwrite_chat_template:
            log.warning("Overwriting the chat template with the default chat template.")
            tokenizer.chat_template = chat_template_mapping[model_family]
        else:
            log.warning("Chat template already exists. Skipping overwriting.")
    return tokenizer
