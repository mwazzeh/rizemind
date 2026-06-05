"""Tests for split_learning_mod — no torch dependency."""

from unittest.mock import MagicMock

from flwr.common import Code, Parameters, Status
from flwr.common.constant import MessageType
from flwr.common.recorddict_compat import fitins_to_recorddict, fitres_to_recorddict
from flwr.common.typing import FitIns, FitRes
from rizemind.split_learning.mod import (
    SL_PHASE_BACKWARD,
    SL_PHASE_FORWARD,
    SL_PHASE_KEY,
    split_learning_mod,
)
from rizemind.split_learning.serialization import _SL_TENSOR_TYPE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUMPY_PARAMS = Parameters(tensors=[b"\x01"], tensor_type="numpy.ndarray")
_SL_PARAMS = Parameters(tensors=[b"\x02"], tensor_type=_SL_TENSOR_TYPE)
_OK_STATUS = Status(code=Code.OK, message="")


def _fit_reply(params: Parameters = _NUMPY_PARAMS) -> MagicMock:
    fit_res = FitRes(
        status=_OK_STATUS,
        parameters=params,
        num_examples=1,
        metrics={},
    )
    reply = MagicMock()
    reply.content = fitres_to_recorddict(fit_res, keep_input=False)
    return reply


def _train_msg(params: Parameters) -> MagicMock:
    fit_ins = FitIns(params, {})
    msg = MagicMock()
    msg.metadata.message_type = MessageType.TRAIN
    msg.content = fitins_to_recorddict(fit_ins, keep_input=False)
    return msg


def _non_train_msg() -> MagicMock:
    msg = MagicMock()
    msg.metadata.message_type = "evaluate"
    return msg


# ---------------------------------------------------------------------------
# Non-TRAIN messages
# ---------------------------------------------------------------------------


def test_non_train_passes_through_without_calling_nested_logic():
    msg = _non_train_msg()
    ctx = MagicMock()
    sentinel = MagicMock()
    call_next = MagicMock(return_value=sentinel)

    result = split_learning_mod(msg, ctx, call_next)

    call_next.assert_called_once_with(msg, ctx)
    assert result is sentinel


# ---------------------------------------------------------------------------
# Forward round — phase detection and config injection
# ---------------------------------------------------------------------------


def test_forward_round_injects_sl_phase_forward():
    captured_msg = {}

    def call_next(msg, ctx):
        from flwr.common.recorddict_compat import recorddict_to_fitins

        fit_ins = recorddict_to_fitins(msg.content, keep_input=True)
        captured_msg["config"] = dict(fit_ins.config)
        return _fit_reply()

    split_learning_mod(_train_msg(_NUMPY_PARAMS), MagicMock(), call_next)

    assert captured_msg["config"][SL_PHASE_KEY] == SL_PHASE_FORWARD


def test_forward_round_relabels_reply_tensor_type():
    call_next = MagicMock(return_value=_fit_reply(_NUMPY_PARAMS))

    result = split_learning_mod(_train_msg(_NUMPY_PARAMS), MagicMock(), call_next)

    from flwr.common.recorddict_compat import recorddict_to_fitres

    fit_res = recorddict_to_fitres(result.content, keep_input=True)
    assert fit_res.parameters.tensor_type == _SL_TENSOR_TYPE


def test_forward_round_preserves_payload_bytes():
    # Capture bytes separately: fitres_to_recorddict(keep_input=False) mutates
    # the tensors list in-place, so we must not compare against the list later.
    payload = b"\xab\xcd"
    reply_params = Parameters(tensors=[payload], tensor_type="numpy.ndarray")
    call_next = MagicMock(return_value=_fit_reply(reply_params))

    result = split_learning_mod(_train_msg(_NUMPY_PARAMS), MagicMock(), call_next)

    from flwr.common.recorddict_compat import recorddict_to_fitres

    fit_res = recorddict_to_fitres(result.content, keep_input=True)
    assert fit_res.parameters.tensors == [payload]


# ---------------------------------------------------------------------------
# Backward round — phase detection and passthrough
# ---------------------------------------------------------------------------


def test_backward_round_injects_sl_phase_backward():
    captured_msg = {}

    def call_next(msg, ctx):
        from flwr.common.recorddict_compat import recorddict_to_fitins

        fit_ins = recorddict_to_fitins(msg.content, keep_input=True)
        captured_msg["config"] = dict(fit_ins.config)
        return _fit_reply()

    split_learning_mod(_train_msg(_SL_PARAMS), MagicMock(), call_next)

    assert captured_msg["config"][SL_PHASE_KEY] == SL_PHASE_BACKWARD


def test_backward_round_does_not_relabel_tensor_type():
    weight_reply = _fit_reply(_NUMPY_PARAMS)
    call_next = MagicMock(return_value=weight_reply)

    result = split_learning_mod(_train_msg(_SL_PARAMS), MagicMock(), call_next)

    from flwr.common.recorddict_compat import recorddict_to_fitres

    fit_res = recorddict_to_fitres(result.content, keep_input=True)
    assert fit_res.parameters.tensor_type == "numpy.ndarray"


def test_backward_round_returns_call_next_reply():
    reply = _fit_reply()
    call_next = MagicMock(return_value=reply)

    result = split_learning_mod(_train_msg(_SL_PARAMS), MagicMock(), call_next)

    assert result is reply


# ---------------------------------------------------------------------------
# call_next is always called exactly once for TRAIN messages
# ---------------------------------------------------------------------------


def test_train_message_always_calls_call_next_once():
    for params in [_NUMPY_PARAMS, _SL_PARAMS]:
        call_next = MagicMock(return_value=_fit_reply())
        split_learning_mod(_train_msg(params), MagicMock(), call_next)
        call_next.assert_called_once()
