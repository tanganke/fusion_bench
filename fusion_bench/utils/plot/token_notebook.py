import numpy as np
from IPython.display import HTML, display


def create_color_style():
    return """
    <style>
        .token-container { font-family: monospace; white-space: pre; }
        .attention { background-color: #90EE90; }  /* Light green */
        .label { background-color: #FFB6C6; }      /* Light red */
        .token { color: #0066cc; }                 /* Blue */
        .stats { font-weight: bold; }
    </style>
    """


def escape_special_chars(text):
    """Convert special characters to their string representation"""
    return (
        text.replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace(" ", "␣")
    )  # Optional: show spaces with visible character


def visualize_tokens_html(input_ids, attention_mask, labels, tokenizer):
    """
    Visualize model inputs using HTML colored text representation for Jupyter Notebook
    with special characters shown as strings
    """
    # Convert to numpy if tensors
    attention_mask = np.array(attention_mask).flatten()
    labels = np.array(labels).flatten()
    input_ids = np.array(input_ids).flatten()

    # Decode tokens and escape special characters
    tokens = [escape_special_chars(tokenizer.decode(id_)) for id_ in input_ids]

    # Create HTML output
    html_output = [create_color_style()]

    # Header
    html_output.append("<h3>**Token Visualization**</h3>")

    # Legend
    html_output.append(
        """
    <div style='margin: 10px 0;'>
        <strong>Legend:</strong><br>
        <span class='attention'>&nbsp;&nbsp;&nbsp;&nbsp;</span> Active Attention<br>
        <span class='label'>&nbsp;&nbsp;&nbsp;&nbsp;</span> Label Present<br>
        <span class='token'>Text</span> Token Text<br>
        Special Characters: \\n (newline), \\t (tab), ␣ (space)
    </div>
    """
    )

    # Token alignment
    html_output.append("<strong>Token Alignment:</strong>")
    html_output.append("<div class='token-container'>")

    # Calculate maximum token length for better alignment
    max_token_len = max(len(str(token)) for token in tokens)

    for i, (input_id, token, mask, label) in enumerate(
        zip(input_ids, tokens, attention_mask, labels)
    ):
        # Pad token for alignment
        token_text = f"{token:{max_token_len}s}"

        # Create classes for styling
        classes = []
        if mask == 1:
            classes.append("attention")
        if label != -100 and label != 0:
            classes.append("label")

        class_str = f"class='{' '.join(classes)}'" if classes else ""

        # Create the line
        line = f"Position {i:3d}: <span {class_str}><span class='token'>{token_text}</span></span> "
        line += (
            f"(Mask: {int(mask)}, Label: {int(label)}, Inpu_id: {int(input_id)})<br>"
        )
        html_output.append(line)

    html_output.append("</div>")

    # Statistics
    html_output.append(
        """
    <div class='stats' style='margin-top: 10px;'>
        Statistics:<br>
        Total tokens: {}<br>
        Active attention tokens: {}<br>
        Labeled tokens: {}<br>
    </div>
    """.format(
            len(tokens), attention_mask.sum(), sum(labels != -100)
        )
    )

    # Display the HTML
    display(HTML("".join(html_output)))


# Example usage:
"""
from transformers import AutoTokenizer
import torch

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Sample input with special characters
text = "Hello,\nhow are\tyou?"
inputs = tokenizer(text, return_tensors='pt')
labels = torch.zeros_like(inputs['input_ids'])  # dummy labels

visualize_tokens_html(
    inputs['attention_mask'][0],
    labels[0],
    inputs['input_ids'][0],
    tokenizer
)
"""
