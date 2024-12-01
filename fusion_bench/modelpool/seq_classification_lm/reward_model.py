from transformers import AutoModelForSequenceClassification


def create_reward_model_from_pretrained(pretrained_model_name_or_path: str, **kwargs):
    """
    Create a reward model for reward modeling (RLHF).

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.
        **kwargs: Additional keyword arguments passed to the model class.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path, num_labels=1, **kwargs
    )
    return model
