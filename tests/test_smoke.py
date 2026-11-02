"""Teste de fumaça: o ambiente treina algo pequeno e é reprodutível."""

import torch
from torch import nn

from dlcurso.env import get_device, seed_everything


def _train_tiny(seed: int, steps: int = 50) -> list[float]:
    seed_everything(seed)
    device = get_device()
    x = torch.randn(256, 16, device=device)
    y = x @ torch.randn(16, 1, device=device) + 0.1 * torch.randn(256, 1, device=device)
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_loss_decreases():
    losses = _train_tiny(seed=0)
    assert losses[-1] < 0.5 * losses[0]


def test_same_seed_same_result():
    assert _train_tiny(seed=1, steps=5) == _train_tiny(seed=1, steps=5)
