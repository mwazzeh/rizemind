from collections.abc import Callable
from logging import DEBUG, INFO

from flwr.common.logger import log
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from rizemind.split_learning.config import SplitLearningConfig
from rizemind.split_learning.serialization import _SL_TENSOR_TYPE

_FORWARD_PHASE = "forward"
_BACKWARD_PHASE = "backward"


class SplitLearningStrategy(Strategy):
    """Flower strategy for split learning.

    Each training step spans ``config.num_rounds_per_step`` Flower rounds
    (default: 2):

    - **Forward round** — the server sends its current model parameters to
      clients. Each client runs its local layers up to the cut point and returns
      the activations packed as ``Parameters`` with
      ``tensor_type == "split_learning.activation"``.

    - **Backward round** — after computing the gradient at the cut point
      (optionally via ``server_backward_fn``), the server sends per-client
      gradients back. Each client completes backpropagation and returns its
      updated weights.

    Phase detection is done by inspecting ``FitRes.parameters.tensor_type`` of
    the first result in ``aggregate_fit``. Activation results trigger the
    backward round; anything else is treated as updated weights and resets the
    cycle.

    Attributes:
        strategy: Base Flower strategy used for parameter initialisation,
            weight-update aggregation, and evaluation.
        config: Split-learning configuration (cut layer, rounds per step).
        server_backward_fn: Optional callable that receives the activation store
            ``{cid: Parameters}`` and returns a gradient store ``{cid: Parameters}``.
            When ``None`` the gradient store is left empty; ``mod.py`` or the
            example's server loop is expected to populate it externally.
    """

    strategy: Strategy
    config: SplitLearningConfig
    server_backward_fn: Callable[[dict[str, Parameters]], dict[str, Parameters]] | None

    def __init__(
        self,
        strategy: Strategy,
        config: SplitLearningConfig,
        server_backward_fn: (
            Callable[[dict[str, Parameters]], dict[str, Parameters]] | None
        ) = None,
    ) -> None:
        """Initialise the split-learning strategy.

        Args:
            strategy: Base Flower strategy for initialisation, aggregation, and
                evaluation of weight updates.
            config: Split-learning configuration.
            server_backward_fn: Optional callable
                ``(activation_store) -> gradient_store`` invoked during
                ``aggregate_fit`` after storing client activations.  When
                ``None`` the gradient store remains empty until populated
                by other means.
        """
        super().__init__()
        self.strategy = strategy
        self.config = config
        self.server_backward_fn = server_backward_fn
        self._phase: str = _FORWARD_PHASE
        self._activation_store: dict[str, Parameters] = {}
        self._gradient_store: dict[str, Parameters] = {}
        self._server_parameters: Parameters | None = None
        log(DEBUG, "SplitLearningStrategy: initialised (phase=%s)", self._phase)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialise server-side model parameters via the base strategy.

        Args:
            client_manager: Flower client manager.

        Returns:
            Initial model parameters, or ``None``.
        """
        self._server_parameters = self.strategy.initialize_parameters(client_manager)
        return self._server_parameters

    # ------------------------------------------------------------------
    # Fit (training)
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Build fit instructions for the current phase.

        - **Forward phase**: save ``parameters`` as the current server state,
          then delegate to the base strategy so clients receive the server's
          model weights.
        - **Backward phase**: send each client in ``_gradient_store`` its
          per-client gradient tensor as a ``FitIns``.  Clients absent from the
          store (i.e. they did not contribute an activation in the preceding
          forward round) are silently skipped.

        Args:
            server_round: Current Flower round number.
            parameters: Current global model parameters (used in forward phase
                only).
            client_manager: Flower client manager.

        Returns:
            List of ``(ClientProxy, FitIns)`` pairs.
        """
        if self._phase == _BACKWARD_PHASE:
            log(DEBUG, "configure_fit: backward phase — sending per-client gradients")
            available = client_manager.all()
            instructions = [
                (client, FitIns(self._gradient_store[cid], {}))
                for cid, client in available.items()
                if cid in self._gradient_store
            ]
            log(
                INFO,
                "configure_fit: backward phase — %d client(s) with gradients",
                len(instructions),
            )
            return instructions

        log(DEBUG, "configure_fit: forward phase — delegating to base strategy")
        self._server_parameters = parameters
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results and advance to the next round phase.

        Inspects the ``tensor_type`` of the first result to determine the phase:

        - **Activation results** (``tensor_type == "split_learning.activation"``)
          — forward round is complete.  Store each client's activations by
          ``cid``, invoke ``server_backward_fn`` if provided to populate the
          gradient store, and transition to the backward phase.  Returns the
          current server parameters unchanged so Flower's global model is not
          modified.
        - **Weight-update results** — backward round is complete.  Clear both
          stores, transition back to the forward phase, and delegate aggregation
          to the base strategy.

        Args:
            server_round: Current Flower round number.
            results: List of ``(ClientProxy, FitRes)`` from clients.
            failures: List of failed client results or exceptions.

        Returns:
            Tuple of ``(parameters, metrics)``.  During forward rounds this is
            ``(server_parameters, {})``.  During backward rounds this is
            whatever the base strategy returns.
        """
        if self._is_activation_results(results):
            log(
                DEBUG,
                "aggregate_fit: forward round — storing activations from %d client(s)",
                len(results),
            )
            for client, res in results:
                self._activation_store[client.cid] = res.parameters

            if self.server_backward_fn is not None:
                log(DEBUG, "aggregate_fit: invoking server_backward_fn")
                self._gradient_store = self.server_backward_fn(
                    dict(self._activation_store)
                )

            self._phase = _BACKWARD_PHASE
            log(INFO, "aggregate_fit: transitioning to backward phase")
            return self._server_parameters, {}

        log(
            DEBUG,
            "aggregate_fit: backward round — delegating to base strategy",
        )
        self._activation_store.clear()
        self._gradient_store.clear()
        self._phase = _FORWARD_PHASE
        log(INFO, "aggregate_fit: transitioning to forward phase")
        return self.strategy.aggregate_fit(server_round, results, failures)

    # ------------------------------------------------------------------
    # Evaluate (fully delegated)
    # ------------------------------------------------------------------

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Delegate evaluation configuration to the base strategy.

        Args:
            server_round: Current Flower round number.
            parameters: Current global model parameters.
            client_manager: Flower client manager.

        Returns:
            Evaluation instructions from the base strategy.
        """
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Delegate evaluation aggregation to the base strategy.

        Args:
            server_round: Current Flower round number.
            results: List of client evaluation results.
            failures: List of failed evaluations.

        Returns:
            Aggregated evaluation result from the base strategy.
        """
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Delegate server-side evaluation to the base strategy.

        Args:
            server_round: Current Flower round number.
            parameters: Current global model parameters.

        Returns:
            Evaluation result from the base strategy, or ``None``.
        """
        return self.strategy.evaluate(server_round, parameters)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_activation_results(results: list[tuple[ClientProxy, FitRes]]) -> bool:
        """Return True if the first result carries split-learning activations.

        Args:
            results: List of ``(ClientProxy, FitRes)`` pairs.

        Returns:
            ``True`` when the first ``FitRes`` has
            ``tensor_type == "split_learning.activation"``, ``False`` otherwise
            (including when ``results`` is empty).
        """
        if not results:
            return False
        _, first_res = results[0]
        return first_res.parameters.tensor_type == _SL_TENSOR_TYPE
