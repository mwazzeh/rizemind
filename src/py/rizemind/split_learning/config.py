from pydantic import Field, field_validator

from rizemind.configuration.base_config import BaseConfig


class SplitLearningConfig(BaseConfig):
    """Configuration for split learning.

    Specifies where the model is split and how many Flower rounds make up one
    training step. Each step uses two rounds by default: the client sends
    activations in the first round, and the server returns the gradient in
    the second so the client can finish backpropagation.

    Attributes:
        cut_layer: Zero-based index of the last layer executed on the client.
            Layer ``cut_layer + 1`` is the first layer executed on the server.
        num_rounds_per_step: Number of Flower rounds per training step.
            Must be >= 1. Defaults to 2.
    """

    cut_layer: int = Field(
        ...,
        description="Zero-based index of the last client-side layer.",
    )
    num_rounds_per_step: int = Field(
        default=2,
        description="Flower rounds per split-learning training step. Default: 2.",
    )

    @field_validator("cut_layer")
    @classmethod
    def _cut_layer_non_negative(cls, value: int) -> int:
        """Validate that cut_layer is non-negative.

        Args:
            value: The cut_layer value to validate.

        Returns:
            The validated cut_layer value.

        Raises:
            ValueError: If cut_layer is negative.
        """
        if value < 0:
            raise ValueError(f"cut_layer must be >= 0, got {value}")
        return value

    @field_validator("num_rounds_per_step")
    @classmethod
    def _num_rounds_at_least_one(cls, value: int) -> int:
        """Validate that num_rounds_per_step is at least 1.

        Args:
            value: The num_rounds_per_step value to validate.

        Returns:
            The validated num_rounds_per_step value.

        Raises:
            ValueError: If num_rounds_per_step is less than 1.
        """
        if value < 1:
            raise ValueError(f"num_rounds_per_step must be >= 1, got {value}")
        return value
