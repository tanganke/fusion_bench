# Import necessary modules
import fnmatch
import time
from collections import defaultdict

import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0")):
    """
    Evaluate wikitext-2 perplexity (ppl) on a specified model and tokenizer.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        device: The device to run the evaluation on.

    Returns:
        ppl_test: The perplexity of the model on the test dataset.
    """
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset during training.

    Args:
        model: The model to evaluate.
        trainloader: The training data loader.
        bs: Batch size.
        device: The device to run the evaluation on.

    Returns:
        ppl: The perplexity of the model on the training dataset.
    """
    # Calculate number of samples
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs: int = 1, device=None):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset.

    Args:
        model: The model to evaluate.
        testenc: The test data encoder.
        bs: Batch size.
        device: The device to run the evaluation on.

    Returns:
        ppl: The perplexity of the model on the test dataset.
    """
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen) : (j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(
    model_name,
    model,
    tokenizer,
    task_list=[
        "boolq",
        "rte",
        "hellaswag",
        "winogrande",
        "arc_challenge",
        "arc_easy",
        "openbookqa",
    ],
    num_fewshot=0,
    use_accelerate=False,
    add_special_tokens=False,
):
    """
    Evaluate the model on a list of tasks in a zero-shot setting.

    Args:
        model_name: The name of the model.
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        task_list: List of tasks to evaluate on.
        num_fewshot: Number of few-shot examples.
        use_accelerate: Whether to use the accelerate library.
        add_special_tokens: Whether to add special tokens.

    Returns:
        results: The evaluation results.
    """
    from lm_eval import evaluator, tasks

    def pattern_match(patterns, source_list):
        """
        Match patterns in the source list.

        Args:
            patterns: List of patterns to match.
            source_list: List of source items.

        Returns:
            task_names: List of matched task names.
        """
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = (
            f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
        )
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
    )

    return results
