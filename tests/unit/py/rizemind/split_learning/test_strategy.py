from unittest.mock import MagicMock, Mock

from flwr.common import Code, Parameters, Status
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.serialization import _SL_TENSOR_TYPE
from rizemind.split_learning.strategy import (
    _BACKWARD_PHASE,
    _FORWARD_PHASE,
    SplitLearningStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG = SplitLearningConfig(cut_layer=2)

_SERVER_PARAMS = Parameters(tensors=[b"\xff"], tensor_type="numpy.ndarray")
_GRADIENT_PARAMS = Parameters(tensors=[b"\xaa"], tensor_type=_SL_TENSOR_TYPE)
_WEIGHT_PARAMS = Parameters(tensors=[b"\xbb"], tensor_type="numpy.ndarray")


def _make_client(cid: str) -> ClientProxy:
    client = Mock(spec=ClientProxy)
    client.cid = cid
    return client


def _activation_res() -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=Parameters(tensors=[b"\x01"], tensor_type=_SL_TENSOR_TYPE),
        num_examples=1,
        metrics={},
    )


def _weight_res() -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=_WEIGHT_PARAMS,
        num_examples=1,
        metrics={},
    )


def _make_strategy(server_backward_fn=None) -> SplitLearningStrategy:
    base = MagicMock()
    return SplitLearningStrategy(
        strategy=base,
        config=_CONFIG,
        server_backward_fn=server_backward_fn,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_initial_phase_is_forward():
    strat = _make_strategy()
    assert strat._phase == _FORWARD_PHASE


def test_initial_stores_are_empty():
    strat = _make_strategy()
    assert strat._activation_store == {}
    assert strat._gradient_store == {}


def test_initial_server_parameters_is_none():
    strat = _make_strategy()
    assert strat._server_parameters is None


# ---------------------------------------------------------------------------
# initialize_parameters
# ---------------------------------------------------------------------------


def test_initialize_parameters_delegates():
    strat = _make_strategy()
    strat.strategy.initialize_parameters.return_value = _SERVER_PARAMS
    cm = MagicMock()

    result = strat.initialize_parameters(cm)

    strat.strategy.initialize_parameters.assert_called_once_with(cm)
    assert result is _SERVER_PARAMS


def test_initialize_parameters_stores_result():
    strat = _make_strategy()
    strat.strategy.initialize_parameters.return_value = _SERVER_PARAMS
    strat.initialize_parameters(MagicMock())
    assert strat._server_parameters is _SERVER_PARAMS


# ---------------------------------------------------------------------------
# configure_fit — forward phase
# ---------------------------------------------------------------------------


def test_configure_fit_forward_delegates_to_base():
    strat = _make_strategy()
    cm = MagicMock()
    expected = [(Mock(), MagicMock())]
    strat.strategy.configure_fit.return_value = expected

    result = strat.configure_fit(1, _SERVER_PARAMS, cm)

    strat.strategy.configure_fit.assert_called_once_with(1, _SERVER_PARAMS, cm)
    assert result is expected


def test_configure_fit_forward_saves_server_parameters():
    strat = _make_strategy()
    strat.strategy.configure_fit.return_value = []

    strat.configure_fit(1, _SERVER_PARAMS, MagicMock())

    assert strat._server_parameters is _SERVER_PARAMS


# ---------------------------------------------------------------------------
# configure_fit — backward phase
# ---------------------------------------------------------------------------


def test_configure_fit_backward_returns_per_client_instructions():
    strat = _make_strategy()
    strat._phase = _BACKWARD_PHASE

    c1, c2 = _make_client("a"), _make_client("b")
    strat._gradient_store = {"a": _GRADIENT_PARAMS, "b": _GRADIENT_PARAMS}
    cm = MagicMock()
    cm.all.return_value = {"a": c1, "b": c2}

    result = strat.configure_fit(2, _SERVER_PARAMS, cm)

    cids = {client.cid for client, _ in result}
    assert cids == {"a", "b"}
    assert all(fit_ins.parameters is _GRADIENT_PARAMS for _, fit_ins in result)


def test_configure_fit_backward_skips_clients_without_gradients():
    strat = _make_strategy()
    strat._phase = _BACKWARD_PHASE

    c1, c2 = _make_client("a"), _make_client("b")
    strat._gradient_store = {"a": _GRADIENT_PARAMS}  # "b" has no gradient
    cm = MagicMock()
    cm.all.return_value = {"a": c1, "b": c2}

    result = strat.configure_fit(2, _SERVER_PARAMS, cm)

    assert len(result) == 1
    assert result[0][0].cid == "a"


def test_configure_fit_backward_does_not_call_base_strategy():
    strat = _make_strategy()
    strat._phase = _BACKWARD_PHASE
    strat._gradient_store = {}
    cm = MagicMock()
    cm.all.return_value = {}

    strat.configure_fit(2, _SERVER_PARAMS, cm)

    strat.strategy.configure_fit.assert_not_called()


# ---------------------------------------------------------------------------
# aggregate_fit — forward round (activation results)
# ---------------------------------------------------------------------------


def test_aggregate_fit_forward_stores_activations_by_cid():
    strat = _make_strategy()
    strat._server_parameters = _SERVER_PARAMS
    c1, c2 = _make_client("x"), _make_client("y")
    results = [(c1, _activation_res()), (c2, _activation_res())]

    strat.aggregate_fit(1, results, [])

    assert set(strat._activation_store.keys()) == {"x", "y"}


def test_aggregate_fit_forward_calls_server_backward_fn():
    received: list[dict] = []

    def backward_fn(store):
        received.append(dict(store))
        return {"x": _GRADIENT_PARAMS}

    strat = _make_strategy(server_backward_fn=backward_fn)
    strat._server_parameters = _SERVER_PARAMS
    c = _make_client("x")
    strat.aggregate_fit(1, [(c, _activation_res())], [])

    assert len(received) == 1
    assert "x" in received[0]


def test_aggregate_fit_forward_populates_gradient_store_from_fn():
    def backward_fn(store):
        return {cid: _GRADIENT_PARAMS for cid in store}

    strat = _make_strategy(server_backward_fn=backward_fn)
    strat._server_parameters = _SERVER_PARAMS
    c = _make_client("x")
    strat.aggregate_fit(1, [(c, _activation_res())], [])

    assert "x" in strat._gradient_store


def test_aggregate_fit_forward_transitions_to_backward():
    strat = _make_strategy()
    strat._server_parameters = _SERVER_PARAMS
    c = _make_client("x")
    strat.aggregate_fit(1, [(c, _activation_res())], [])
    assert strat._phase == _BACKWARD_PHASE


def test_aggregate_fit_forward_returns_server_parameters():
    strat = _make_strategy()
    strat._server_parameters = _SERVER_PARAMS
    c = _make_client("x")
    params, metrics = strat.aggregate_fit(1, [(c, _activation_res())], [])
    assert params is _SERVER_PARAMS
    assert metrics == {}


def test_aggregate_fit_forward_does_not_call_base_strategy():
    strat = _make_strategy()
    strat._server_parameters = _SERVER_PARAMS
    c = _make_client("x")
    strat.aggregate_fit(1, [(c, _activation_res())], [])
    strat.strategy.aggregate_fit.assert_not_called()


# ---------------------------------------------------------------------------
# aggregate_fit — backward round (weight-update results)
# ---------------------------------------------------------------------------


def test_aggregate_fit_backward_delegates_to_base():
    strat = _make_strategy()
    strat._phase = _BACKWARD_PHASE
    strat._activation_store = {"x": _GRADIENT_PARAMS}
    strat._gradient_store = {"x": _GRADIENT_PARAMS}

    c = _make_client("x")
    failures: list = []
    results = [(c, _weight_res())]
    strat.strategy.aggregate_fit.return_value = (_WEIGHT_PARAMS, {"loss": 0.5})

    returned = strat.aggregate_fit(2, results, failures)

    strat.strategy.aggregate_fit.assert_called_once_with(2, results, failures)
    assert returned == (_WEIGHT_PARAMS, {"loss": 0.5})


def test_aggregate_fit_backward_transitions_to_forward():
    strat = _make_strategy()
    strat._phase = _BACKWARD_PHASE
    strat.strategy.aggregate_fit.return_value = (None, {})

    c = _make_client("x")
    strat.aggregate_fit(2, [(c, _weight_res())], [])

    assert strat._phase == _FORWARD_PHASE


def test_aggregate_fit_backward_clears_stores():
    strat = _make_strategy()
    strat._phase = _BACKWARD_PHASE
    strat._activation_store = {"x": _GRADIENT_PARAMS}
    strat._gradient_store = {"x": _GRADIENT_PARAMS}
    strat.strategy.aggregate_fit.return_value = (None, {})

    c = _make_client("x")
    strat.aggregate_fit(2, [(c, _weight_res())], [])

    assert strat._activation_store == {}
    assert strat._gradient_store == {}


# ---------------------------------------------------------------------------
# Empty results edge case
# ---------------------------------------------------------------------------


def test_aggregate_fit_empty_results_treated_as_backward():
    """Empty results list must not raise and must delegate to base strategy."""
    strat = _make_strategy()
    strat.strategy.aggregate_fit.return_value = (None, {})

    strat.aggregate_fit(1, [], [])

    strat.strategy.aggregate_fit.assert_called_once()
    assert strat._phase == _FORWARD_PHASE


# ---------------------------------------------------------------------------
# Evaluation delegation
# ---------------------------------------------------------------------------


def test_configure_evaluate_delegates():
    strat = _make_strategy()
    cm = MagicMock()
    expected = [(Mock(), MagicMock())]
    strat.strategy.configure_evaluate.return_value = expected

    result = strat.configure_evaluate(1, _SERVER_PARAMS, cm)

    strat.strategy.configure_evaluate.assert_called_once_with(1, _SERVER_PARAMS, cm)
    assert result is expected


def test_aggregate_evaluate_delegates():
    strat = _make_strategy()
    expected = (0.42, {"accuracy": 0.9})
    strat.strategy.aggregate_evaluate.return_value = expected

    results, failures = [Mock()], []
    result = strat.aggregate_evaluate(1, results, failures)

    strat.strategy.aggregate_evaluate.assert_called_once_with(1, results, failures)
    assert result is expected


def test_evaluate_delegates():
    strat = _make_strategy()
    expected = (0.1, {})
    strat.strategy.evaluate.return_value = expected

    result = strat.evaluate(1, _SERVER_PARAMS)

    strat.strategy.evaluate.assert_called_once_with(1, _SERVER_PARAMS)
    assert result is expected
