"""Split learning support for Rizemind.

Each training step spans two Flower rounds: a forward round where the client
sends activations to the server, followed by a backward round where the server
returns the gradient so the client can complete backpropagation.
"""

from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.gradient_privacy import (
    GradientPrivacyConfig,
    privatize_joint_gradient,
)
from rizemind.split_learning.label_inference_attack import evaluate_label_inference
from rizemind.split_learning.label_private_strategy import LabelPrivateVerticalStrategy
from rizemind.split_learning.metrics import classification_metrics
from rizemind.split_learning.mod import split_learning_mod
from rizemind.split_learning.seeding import seed_everything
from rizemind.split_learning.serialization import (
    parameters_to_tensor,
    tensor_to_parameters,
)
from rizemind.split_learning.strategy import SplitLearningStrategy
from rizemind.split_learning.telemetry import RunTelemetry, StepTelemetry
from rizemind.split_learning.vertical_strategy import VerticalSplitLearningStrategy

__all__ = [
    "GradientPrivacyConfig",
    "LabelPrivateVerticalStrategy",
    "RunTelemetry",
    "SplitLearningConfig",
    "SplitLearningStrategy",
    "StepTelemetry",
    "VerticalSplitLearningStrategy",
    "classification_metrics",
    "evaluate_label_inference",
    "parameters_to_tensor",
    "privatize_joint_gradient",
    "seed_everything",
    "split_learning_mod",
    "tensor_to_parameters",
]
