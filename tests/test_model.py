"""Tests for BikeNet."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from model import BikeNet


@pytest.mark.parametrize(
    "hidden_size",
    [8, 16, 32],
)
def test_network_output_shape(hidden_size):
    model = BikeNet(
        n_features=8,
        hidden=hidden_size,
    )

    x = torch.randn(
        10,
        8,
    )

    output = model(x)

    assert output.shape == (10,)


def test_optimizer_step_changes_weights():
    model = BikeNet(
        n_features=8,
    )

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    x = torch.randn(
        32,
        8,
    )

    y = torch.randn(32)

    before = [p.detach().clone() for p in model.parameters()]

    prediction = model(x)

    loss = loss_fn(
        prediction,
        y,
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    after = list(model.parameters())

    changed = False

    for old, new in zip(
        before,
        after,
    ):
        if not torch.equal(
            old,
            new,
        ):
            changed = True

    assert changed
