"""Split learning support for Rizemind.

Each training step spans two Flower rounds: a forward round where the client
sends activations to the server, followed by a backward round where the server
returns the gradient so the client can complete backpropagation.
"""

from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.mod import split_learning_mod
from rizemind.split_learning.serialization import (
    parameters_to_tensor,
    tensor_to_parameters,
)
from rizemind.split_learning.strategy import SplitLearningStrategy

__all__ = [
    "SplitLearningConfig",
    "SplitLearningStrategy",
    "parameters_to_tensor",
    "split_learning_mod",
    "tensor_to_parameters",
]
