import numpy as np
import torch
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

# Marker stored in Parameters.tensor_type so the strategy can distinguish
# split-learning payloads from ordinary model-weight Parameters.
_SL_TENSOR_TYPE = "split_learning.activation"


def tensor_to_parameters(tensor: torch.Tensor) -> Parameters:
    """Serialize a single tensor into Flower ``Parameters``.

    Converts ``tensor`` to a NumPy array and packs it using Flower's standard
    wire format.  The resulting ``Parameters.tensor_type`` is set to
    ``"split_learning.activation"`` so callers can distinguish SL payloads
    from ordinary model-weight parameters.

    The tensor is detached from the autograd graph and moved to CPU before
    serialization; the caller retains ownership of the original tensor.

    Args:
        tensor: Activation or gradient tensor produced at the cut layer.
            May have any dtype and may require gradients.

    Returns:
        A ``Parameters`` object carrying the serialized tensor.
    """
    ndarray: np.ndarray = tensor.detach().cpu().numpy()
    params = ndarrays_to_parameters([ndarray])
    # Override the default tensor_type to mark this as an SL payload.
    params = Parameters(tensors=params.tensors, tensor_type=_SL_TENSOR_TYPE)
    return params


def parameters_to_tensor(
    parameters: Parameters,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Deserialize Flower ``Parameters`` back into a single tensor.

    Unpacks the first (and only expected) array from ``parameters`` and
    converts it to a ``torch.Tensor`` on CPU.  Pass ``requires_grad=True``
    when the receiving side needs to continue backpropagation through the
    reconstructed tensor (e.g. the client applying a gradient from the server).

    Args:
        parameters: A ``Parameters`` object produced by
            :func:`tensor_to_parameters`.
        requires_grad: If ``True``, the returned tensor participates in
            autograd. Pass ``True`` on the client when applying a gradient
            received from the server. Defaults to ``False``.

    Returns:
        The deserialized tensor on CPU.

    Raises:
        ValueError: If ``parameters`` contains zero or more than one array,
            since this function expects exactly one serialized tensor.
    """
    ndarrays = parameters_to_ndarrays(parameters)
    if len(ndarrays) != 1:
        raise ValueError(
            f"parameters_to_tensor expects exactly 1 array, got {len(ndarrays)}"
        )
    tensor = torch.from_numpy(ndarrays[0])
    if requires_grad:
        tensor = tensor.requires_grad_(True)
    return tensor
