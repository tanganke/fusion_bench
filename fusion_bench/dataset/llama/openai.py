import logging
from typing import Dict, List

from datasets import Dataset
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)


def tokenize_messages_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    padding: bool = True,
    system_template: str = "### System: {message}\n",
    user_template: str = "## User: {message}\n",
    assistant_template: str = "## Assistant: {message}\n",
) -> Dataset:
    R"""
    Tokenize dataset with messages format supporting loss calculation flags.

    write a script to tokenizer datasets with the following format:

    Examples:

    ```json
    {
        "messages": [
            {
                "role": "system",
                "content": "XXX",
                "calculate_loss": 0
            },
            {
                "role": "system",
                "content": "XXX",
                "calculate_loss": 0
            },
            {
                "role": "user",
                "content": "XXX",
                "calculate_loss": 0
            },
            {
                "role": "assistant",
                "content": "XXX",
                "calculate_loss": 1
            }
        ],
        "create_info": [
            {
                "date": "20240830",
                "owner": "l00470783",
                "within_source_id": 0,
                "describe": "...",
                "source": [
                    "..."
                ],
                "language": "zh"
            }
        ],
        "feature_info": {
            "domain": "...",
            "tags": [
                "..."
            ]
        },
        "source_file": "..."
    }
    ```

    Args:
        dataset: Input dataset with messages format
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        system_template: Template for system messages
        user_template: Template for user messages
        assistant_template: Template for assistant messages

    Returns:
        Tokenized dataset
    """

    def build_prompt(messages: List[Dict[str, str]]) -> tuple[str, str]:
        """
        Build prompt and get response that needs loss calculation.
        Returns conversation history and the response to calculate loss on.
        """
        history = ""
        response = ""

        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            calculate_loss = message.get("calculate_loss", 0)

            # Build conversation history
            if role == "system":
                history += system_template.format(message=content)
            elif role == "user":
                history += user_template.format(message=content)
            elif role == "assistant":
                if calculate_loss:
                    # If this assistant message needs loss calculation,
                    # save it as response and don't add to history
                    response = content
                else:
                    # Otherwise add to conversation history
                    history += assistant_template.format(message=content)

        return history, response

    def prepare_sample(sample: dict) -> dict:
        # Get conversation history and response
        history, response = build_prompt(sample["messages"])

        # Tokenize prompt and response
        prompt_tokens = tokenizer.encode(history, add_special_tokens=False)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)

        # Create input_ids with EOS token
        input_ids = prompt_tokens + response_tokens + [tokenizer.eos_token_id]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Create labels: -100 for prompt, actual tokens for response
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
