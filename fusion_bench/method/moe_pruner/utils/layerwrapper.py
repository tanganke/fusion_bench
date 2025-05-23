import torch
import torch.nn as nn


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.

    Attributes:
        layer (nn.Linear | nn.Module): The GPT layer to be wrapped.
        dev (torch.device): The device on which the layer's weights are stored.
        rows (int): The number of rows in the layer's weight matrix.
        columns (int): The number of columns in the layer's weight matrix.
        scaler_row (torch.Tensor): A tensor to store the scaler values for each column.
        nsamples (int): The number of samples processed.
        layer_id (int): The ID of the layer.
        layer_name (str): The name of the layer.
    """

    def __init__(self, layer: nn.Linear | nn.Module, layer_id=0, layer_name="none"):
        """
        Initialize the WrappedGPT class.

        Args:
            layer (nn.Linear | nn.Module): The GPT layer to be wrapped.
            layer_id (int, optional): The ID of the layer. Defaults to 0.
            layer_name (str, optional): The name of the layer. Defaults to "none".
        """
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of input and output tensors to the scaler_row.

        Args:
            inp (torch.Tensor): The input tensor.
            out (torch.Tensor): The output tensor.
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
