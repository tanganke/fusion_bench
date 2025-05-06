from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _number_of_samples(routing_weights: List[Tensor]):
    count = 0
    for routing_weight in routing_weights:
        count += routing_weight.size(0)
    return count


class LayerWiseRoutingWeightSaver:
    """
    A hook for saving layer-wise routing weights.
    """

    save_path: Path
    "The path to save the layer-wise routing weights."
    max_num: Optional[int]
    "The maximum number of layer-wise routing weights to save. If None, all routing weights will be saved."
    routing_weights: List[Tensor]
    "The list of layer-wise routing weights."

    def __init__(self, save_path: Path, max_num: Optional[int] = None):
        """
        Args:
            save_path (Path): The path to save the layer-wise routing weights.
            max_num (Optional[int]): The maximum number of layer-wise routing weights to save. If None, all routing weights will be saved.
        """
        self.save_path = save_path
        self.max_num = max_num
        self.routing_weights = []

    def __call__(self, module, input: Tuple[Tensor], output: Tensor):
        assert isinstance(output, Tensor), "Output is expected to be a Tensor"
        # (batch_size, num_tokens, num_experts)
        routing_weights = output.detach().cpu()
        if self.max_num is not None and self.max_num > 0:
            if _number_of_samples(self.routing_weights) > self.max_num:
                return
            elif (
                routing_weights.size(0) + _number_of_samples(self.routing_weights)
                > self.max_num
            ):
                self.routing_weights.append(
                    routing_weights[
                        : self.max_num - _number_of_samples(self.routing_weights)
                    ]
                )
            else:
                self.routing_weights.append(routing_weights)
        else:
            self.routing_weights.append(routing_weights)

    def save_routing_weights(self):
        routing_weights = torch.cat(self.routing_weights, dim=0)
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"Saving routing weights to {self.save_path}. Size: {routing_weights.size()}"
            )
            torch.save(routing_weights, self.save_path)
