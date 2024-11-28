import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_model_inputs(input_ids, attention_mask, labels, tokenizer=None):
    """
    Visualize model inputs: attention mask, labels and input_ids

    Parameters:
    -----------
    attention_mask: numpy array or tensor
        The attention mask array
    labels: numpy array or tensor
        The labels array
    input_ids: numpy array or tensor
        The input ids array
    tokenizer: optional
        The tokenizer object to decode input_ids
    """

    # Convert inputs to numpy if they're tensors
    attention_mask = np.array(attention_mask)
    labels = np.array(labels)
    input_ids = np.array(input_ids)

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

    # Plot attention mask
    sns.heatmap(attention_mask.reshape(1, -1), ax=ax1, cmap="Blues", cbar=True)
    ax1.set_title("**Attention Mask**")
    ax1.set_ylabel("Sequence")

    # Plot labels
    sns.heatmap(labels.reshape(1, -1), ax=ax2, cmap="Reds", cbar=True)
    ax2.set_title("**Labels**")
    ax2.set_ylabel("Sequence")

    # Plot input_ids
    sns.heatmap(input_ids.reshape(1, -1), ax=ax3, cmap="Greens", cbar=True)
    ax3.set_title("**Input IDs**")
    ax3.set_ylabel("Sequence")

    # If tokenizer is provided, add decoded tokens as x-axis labels
    if tokenizer:
        decoded_tokens = [tokenizer.decode(token_id) for token_id in input_ids]
        ax3.set_xticks(np.arange(len(decoded_tokens)) + 0.5)
        ax3.set_xticklabels(decoded_tokens, rotation=45, ha="right")

    plt.tight_layout()
    return fig
