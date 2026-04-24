from logging import DEBUG

from flwr.client.typing import ClientAppCallable
from flwr.common import Context, Message
from flwr.common.constant import MessageType
from flwr.common.logger import log
from flwr.common.recorddict_compat import (
    fitins_to_recorddict,
    fitres_to_recorddict,
    recorddict_to_fitins,
    recorddict_to_fitres,
)
from flwr.common.typing import Parameters

from rizemind.split_learning.serialization import _SL_TENSOR_TYPE

#: Key injected into ``FitIns.config`` so the underlying ``NumPyClient``
#: can detect which phase it is in without inspecting ``Parameters`` directly.
SL_PHASE_KEY = "sl_phase"
SL_PHASE_FORWARD = "forward"
SL_PHASE_BACKWARD = "backward"


def split_learning_mod(
    msg: Message,
    ctx: Context,
    call_next: ClientAppCallable,
) -> Message:
    """Flower client middleware for split learning.

    Intercepts ``TRAIN`` messages to manage the forward/backward round cycle
    without modifying any existing strategy or client code:

    - **Forward round** (server sends model weights,
      ``tensor_type != "split_learning.activation"``):

      1. Injects ``sl_phase = "forward"`` into ``FitIns.config`` so the
         underlying ``NumPyClient.fit`` knows to run a forward pass and
         return activations rather than updated weights.
      2. After ``call_next`` returns, re-tags ``FitRes.parameters.tensor_type``
         to ``"split_learning.activation"`` so ``SplitLearningStrategy``
         recognises the result as activations.

    - **Backward round** (server sends gradients,
      ``tensor_type == "split_learning.activation"``):

      1. Injects ``sl_phase = "backward"`` into ``FitIns.config`` so the
         client knows to run a backward pass and return updated head weights.
      2. Passes the reply through without modification (weights already carry
         the correct ``tensor_type``).

    Non-``TRAIN`` messages pass through unchanged.

    Args:
        msg: Incoming Flower message.
        ctx: Flower context (unused directly; forwarded to ``call_next``).
        call_next: Next callable in the middleware chain.

    Returns:
        The (possibly modified) reply message.
    """
    if msg.metadata.message_type != MessageType.TRAIN:
        return call_next(msg, ctx)

    fit_ins = recorddict_to_fitins(msg.content, keep_input=True)
    is_backward = fit_ins.parameters.tensor_type == _SL_TENSOR_TYPE
    phase = SL_PHASE_BACKWARD if is_backward else SL_PHASE_FORWARD

    log(DEBUG, "split_learning_mod: phase=%s", phase)
    fit_ins.config[SL_PHASE_KEY] = phase
    msg.content = fitins_to_recorddict(fit_ins, keep_input=False)

    reply = call_next(msg, ctx)

    if not is_backward:
        # Forward round: re-tag the client's activation output so that
        # SplitLearningStrategy.aggregate_fit can identify it correctly.
        fit_res = recorddict_to_fitres(reply.content, keep_input=False)
        fit_res.parameters = Parameters(
            tensors=fit_res.parameters.tensors,
            tensor_type=_SL_TENSOR_TYPE,
        )
        reply.content = fitres_to_recorddict(fit_res, keep_input=False)
        log(DEBUG, "split_learning_mod: tagged reply as activation")

    return reply
