# Question Answering

## Key Concepts

### Overlapping Tokens

**Overlapping tokens** are segments of text that are repeated between consecutive chunks when a long text needs to be split into smaller pieces due to model's maximum token limit.

Here's a detailed explanation:

1. Why we need overlapping:
    - When a text is too long for the model's context window (max_length)
    - To maintain continuity and context between chunks
    - To avoid losing information that might be split between chunks

2. Key parameters in the code:
    - max_length: Maximum number of tokens allowed
    - stride: Number of overlapping tokens between chunks
    - return_overflowing_tokens: Tells tokenizer to return multiple chunks
    - truncation="only_second": Only truncates the context, not the question

Let's illustrate with an example:

Suppose we have a text: *"The quick brown fox jumps over the lazy sleeping dog"*.
The tokenization might look like this:

```
Chunk 1: [The quick brown fox jumps over]
                    ↓ overlap ↓
Chunk 2:            [brown fox jumps over the lazy]
                                ↓ overlap ↓
Chunk 3:                        [jumps over the lazy sleeping dog]
```

Real-world example with actual tokens:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

question = "What did the fox do?"
context = "The quick brown fox jumps over the lazy sleeping dog. It was a beautiful sunny day."

tokenized = tokenizer(
    question,
    context,
    max_length=16,
    truncation="only_second",
    return_overflowing_tokens=True,
    stride=4
)

# Print the decoded tokens for each chunk
for encoding in tokenized["input_ids"]:
    print(tokenizer.decode(encoding))
```

### Offset Mapping

**Offset mapping** is a feature that provides the character-level mapping between the original text and the tokenized output. It returns a list of tuples (start, end) where:

- start: starting character position in the original text
- end: ending character position in the original text

Here's a detailed breakdown:

1. Structure of offset_mapping:

    ```python
    [(0, 0),    # [CLS] token - special token, maps to nothing
    (0, 3),     # "how" - maps to characters 0-3 in original text
    (4, 8),     # "many" - maps to characters 4-8
    ...]
    ```

2. Special tokens mapping:

    - [CLS], [SEP], [PAD]: represented as (0, 0)
    - These tokens don't correspond to any actual text in the input

3. Usage example:

    ```python
    # Example showing how to use offset_mapping
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "How many cats?"
    tokenized = tokenizer(text, return_offsets_mapping=True)

    for token_id, offset in zip(tokenized["input_ids"], tokenized["offset_mapping"]):
        token = tokenizer.decode([token_id])
        start, end = offset
        original_text = text[start:end] if start != end else "[SPECIAL]"
        print(f"Token: {token}, Offset: {offset}, Original text: {original_text}")
    ```

Main purposes of offset_mapping:

1. Answer span location:
    - Helps locate exact position of answers in QA tasks
    - Maps token positions back to original text positions

2. Token-text alignment:
    - Enables precise tracking of which parts of original text correspond to which tokens
    - Useful for tasks requiring character-level precision

3. Handling overlapping chunks:
    - Helps maintain correct position information when text is split into chunks
    - Essential for combining predictions from multiple chunks

Common operations with offset_mapping:
```python
# Finding original text for a token
def get_original_text(text, offset):
    start, end = offset
    return text[start:end] if start != end else "[SPECIAL]"

# Finding token position for a text span
def find_token_position(offset_mapping, char_start, char_end):
    for idx, (start, end) in enumerate(offset_mapping):
        if start == char_start and end == char_end:
            return idx
    return None
```

This feature is particularly important in Question Answering tasks where you need to:

- Map predicted token positions back to original text
- Handle answer spans across multiple chunks
- Maintain precise position information for answer extraction

### overflow_to_sample_mapping

`overflow_to_sample_mapping` is an index list that maps each feature in the overflowing tokens back to its original sample. It's particularly useful when processing multiple examples with overflow.

Here's a detailed explanation:

- When a text is split into multiple chunks due to length
- Each chunk needs to be traced back to its original example
- `overflow_to_sample_mapping` provides this tracking mechanism

Here's a comprehensive example:

```python
from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Multiple examples
examples = {
    "question": [
        "What is the capital?",
        "Who won the game?"
    ],
    "context": [
        "Paris is the capital of France. It is known for the Eiffel Tower. The city has many historic monuments." * 5,  # Made longer by repeating
        "The Lakers won the game against the Bulls. It was a close match." * 2
    ]
}

# Tokenize with overflow
tokenized_examples = []
for q, c in zip(examples["question"], examples["context"]):
    tokenized = tokenizer(
        q,
        c,
        max_length=50,  # Small max_length for demonstration
        stride=10,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation="only_second"
    )
    tokenized_examples.append(tokenized)

# Let's see how many chunks each example was split into
for i, tokenized in enumerate(tokenized_examples):
    print(f"\nExample {i}:")
    print(f"Number of chunks: {len(tokenized['input_ids'])}")
    print(f"Overflow to sample mapping: {tokenized.overflow_to_sample_mapping}")
```

This might output something like:

```
Example 0:
Number of chunks: 4
Overflow to sample mapping: [0, 0, 0, 0]  # All chunks belong to first example

Example 1:
Number of chunks: 2
Overflow to sample mapping: [0, 0]  # All chunks belong to first example
```

Practical Use Case:

```python
def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context
    sample_mapping = tokenized_examples.overflow_to_sample_mapping

    # For each feature, we need to know from which example it came from
    for i, sample_idx in enumerate(sample_mapping):
        # Get the example's original question
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_start = sequence_ids.index(1)  # Find where context starts
        
        # Set example ID for this feature
        tokenized_examples[i]["example_id"] = examples["id"][sample_idx]
        
        # Set offset mappings for answer spans
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples[i]["offset_mapping"])
        ]

    return tokenized_examples
```

Key Benefits:

1. Tracking Features: 
    - Maps each feature back to its source example
    - Maintains relationship between chunks and original data

2. Data Processing:
    - Helps in maintaining example-level information
    - Essential for combining predictions from multiple chunks

3. Batch Processing:
    - Enables proper batching of features
    - Maintains data integrity during training

Common Use Pattern:

```python
# Example of using overflow_to_sample_mapping in a training loop
for i, sample_idx in enumerate(tokenized_examples.overflow_to_sample_mapping):
    # Get original example ID
    original_example_id = examples["id"][sample_idx]
    
    # Get original answer
    original_answer = examples["answers"][sample_idx]
    
    # Process feature while maintaining connection to original example
    process_feature(tokenized_examples[i], original_example_id, original_answer)
```

This feature is particularly important in Question Answering tasks where:

- Long contexts need to be split into multiple chunks
- Each chunk needs to be processed separately
- Results need to be combined while maintaining reference to original examples

