from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_split.client import SplitFlowerClient
from torch_split.task import (
    CachedTrainPartition,
    ClientHead,
    ServerTail,
    TensorDictDataset,
    get_weights,
    serialize_ndarrays_to_bytes,
    set_weights,
)


def _zero_weights_like(weights: list[np.ndarray]) -> list[np.ndarray]:
    return [np.zeros_like(weight) for weight in weights]


def test_evaluate_uses_tail_weights_from_config_and_restores_local_models():
    train_images = torch.tensor([[4.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    train_labels = torch.tensor([0, 1], dtype=torch.long)
    train_partition = CachedTrainPartition(
        images=train_images,
        labels=train_labels,
        batch_size=2,
    )
    valloader = DataLoader(
        TensorDictDataset(train_images, train_labels),
        batch_size=2,
        shuffle=False,
    )

    head = ClientHead(input_dim=2, hidden_dim=2)
    tail = ServerTail(hidden_dim=2, num_classes=2)
    original_head = _zero_weights_like(get_weights(head))
    original_tail = _zero_weights_like(get_weights(tail))
    set_weights(head, original_head)
    set_weights(tail, original_tail)

    eval_head = ClientHead(input_dim=2, hidden_dim=2)
    eval_tail = ServerTail(hidden_dim=2, num_classes=2)
    set_weights(
        eval_head,
        [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ],
    )
    set_weights(
        eval_tail,
        [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ],
    )

    ctx = SimpleNamespace(state={}, node_id="node-1")
    client = SplitFlowerClient(
        head=head,
        tail=tail,
        train_partition=train_partition,
        valloader=valloader,
        learning_rate=0.01,
        ctx=ctx,
    )

    loss, num_examples, metrics = client.evaluate(
        get_weights(eval_head),
        {"sl_tail_weights": serialize_ndarrays_to_bytes(get_weights(eval_tail))},
    )

    assert num_examples == 2
    assert loss < 0.2
    assert metrics["accuracy"] == 1.0
    assert all(np.array_equal(a, b) for a, b in zip(get_weights(client.head), original_head))
    assert all(np.array_equal(a, b) for a, b in zip(get_weights(client.tail), original_tail))
