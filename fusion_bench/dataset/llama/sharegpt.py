import logging
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)


def tokenize_sharegpt_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    padding: bool = True,
    system_template: str = "### System: {system}\n\n",
    tools_template: str = "### Tools: {tools}\n\n",
    human_template: str = "### Human: {message}\n",
    assistant_template: str = "### Assistant: {message}\n",
    function_template: str = "### Function Call: {message}\n",
    observation_template: str = "### Observation: {message}\n",
) -> Dataset:
    """
    Tokenize ShareGPT format dataset with support for system prompts, tools, and tool calls.

    Args:
        dataset: Input dataset in ShareGPT format.
        tokenizer: The tokenizer to use.
        max_length: Maximum sequence length.
        padding: Whether to pad the tokenized inputs to `max_length`.
        system_template: Template for system messages.
        tools_template: Template for tool descriptions.
        human_template: Template for human messages.
        assistant_template: Template for assistant responses.
        function_template: Template for function calls.
        observation_template: Template for function observations.

    Returns:
        Tokenized dataset
    """

    def build_conversation(
        conversations: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[List[int], List[int]]:
        """
        Build prompt and response token ids from conversations.
        Returns (prompt_tokens, response_tokens) for the last assistant message.
        """
        # Initialize conversation history
        history = ""

        # Add system prompt if provided
        if system:
            history += system_template.format(system=system.strip())

        # Add tools description if provided
        if tools:
            history += tools_template.format(tools=tools.strip())

        prompt_tokens = []
        response_tokens = []

        for i, message in enumerate(conversations):
            msg_from = message["from"]
            msg_value = message["value"].strip()

            # If this is the last assistant message
            if msg_from == "gpt" and i == len(conversations) - 1:
                # Tokenize the current history as prompt
                prompt_tokens = tokenizer.encode(history, add_special_tokens=False)
                # Tokenize the assistant's message as response
                response_tokens = tokenizer.encode(
                    assistant_template.format(message=msg_value),
                    add_special_tokens=False,
                )
                break

            # Build conversation history
            if msg_from == "human":
                history += human_template.format(message=msg_value)
            elif msg_from == "gpt":
                history += assistant_template.format(message=msg_value)
            elif msg_from == "function_call":
                history += function_template.format(message=msg_value)
            elif msg_from == "observation":
                history += observation_template.format(message=msg_value)
            else:
                log.warning(f"Unkonwn role: {msg_from}")

        return prompt_tokens, response_tokens

    def prepare_sample(sample: dict) -> dict:
        # Get prompt and response tokens
        prompt_tokens, response_tokens = build_conversation(
            conversations=sample["conversations"],
            system=sample.get("system"),  # system prompt is optional
            tools=sample.get("tools"),  # tools description is optional
        )

        # Create input_ids with EOS token
        input_ids = prompt_tokens + response_tokens + [tokenizer.eos_token_id]

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Create labels (-100 for prompt, actual tokens for response)
        labels = (
            [-100] * len(prompt_tokens) + response_tokens + [tokenizer.eos_token_id]
        )

        # Truncate if exceeds max length
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        # Pad if necessary
        if padding:
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids.extend([tokenizer.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)
                labels.extend([-100] * padding_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    if tokenizer.pad_token is None:
        log.warning("Tokenizer does not have a `pad_token`. Set it the `eos_token`.")
        tokenizer.pad_token = tokenizer.eos_token
    # Process the dataset
    tokenized_dataset = dataset.map(
        prepare_sample, remove_columns=dataset.column_names, desc="Tokenizing dataset"
    )

    return tokenized_dataset
