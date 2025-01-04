# https://huggingface.co/datasets/tanganke/emnist_mnist
classnames = [str(i) for i in range(10)]
templates = [
    lambda c: f'a photo of the number: "{c}".',
]
