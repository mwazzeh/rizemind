import pytest

torch = pytest.importorskip("torch")

from flwr.common import Parameters  # noqa: E402
from rizemind.split_learning.serialization import (  # noqa: E402
    _SL_TENSOR_TYPE,
    parameters_to_tensor,
    tensor_to_parameters,
)

# ---------------------------------------------------------------------------
# tensor_to_parameters — forward direction
# ---------------------------------------------------------------------------


def test_returns_parameters_instance():
    t = torch.tensor([1.0, 2.0, 3.0])
    result = tensor_to_parameters(t)
    assert isinstance(result, Parameters)


def test_tensor_type_tag():
    t = torch.zeros(4)
    result = tensor_to_parameters(t)
    assert result.tensor_type == _SL_TENSOR_TYPE


def test_tensors_list_has_one_entry():
    t = torch.ones(2, 3)
    result = tensor_to_parameters(t)
    assert len(result.tensors) == 1


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.int32, torch.int64],
)
def test_dtype_preserved_through_round_trip(dtype):
    t = torch.arange(6, dtype=dtype).reshape(2, 3)
    params = tensor_to_parameters(t)
    rt = parameters_to_tensor(params)
    assert rt.dtype == t.dtype


def test_values_preserved_through_round_trip():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    params = tensor_to_parameters(t)
    rt = parameters_to_tensor(params)
    assert torch.equal(rt, t)


def test_shape_preserved_through_round_trip():
    t = torch.randn(5, 8, 3)
    params = tensor_to_parameters(t)
    rt = parameters_to_tensor(params)
    assert rt.shape == t.shape


def test_detaches_grad_tensor():
    """Tensor with requires_grad must be serializable without error."""
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    result = tensor_to_parameters(t)
    assert len(result.tensors) == 1


def test_gradient_tensor_values_correct():
    t = torch.tensor([5.0, 6.0], requires_grad=True)
    params = tensor_to_parameters(t)
    rt = parameters_to_tensor(params)
    assert torch.equal(rt, t.detach())


# ---------------------------------------------------------------------------
# parameters_to_tensor — reverse direction
# ---------------------------------------------------------------------------


def test_no_grad_by_default():
    t = torch.randn(3)
    params = tensor_to_parameters(t)
    rt = parameters_to_tensor(params)
    assert not rt.requires_grad


def test_requires_grad_flag():
    t = torch.randn(3)
    params = tensor_to_parameters(t)
    rt = parameters_to_tensor(params, requires_grad=True)
    assert rt.requires_grad


def test_result_on_cpu():
    t = torch.tensor([1.0, 2.0])
    params = tensor_to_parameters(t)
    rt = parameters_to_tensor(params)
    assert rt.device.type == "cpu"


def test_raises_on_empty_parameters():
    bad = Parameters(tensors=[], tensor_type=_SL_TENSOR_TYPE)
    with pytest.raises(ValueError, match="exactly 1 array"):
        parameters_to_tensor(bad)


def test_raises_on_multiple_tensors():
    from flwr.common import ndarrays_to_parameters

    multi = ndarrays_to_parameters(
        [
            __import__("numpy").array([1.0]),
            __import__("numpy").array([2.0]),
        ]
    )
    bad = Parameters(tensors=multi.tensors, tensor_type=_SL_TENSOR_TYPE)
    with pytest.raises(ValueError, match="exactly 1 array"):
        parameters_to_tensor(bad)
