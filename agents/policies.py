"""Tiny actor-critic MLPs for independent PPO (CPU)."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 8, n_actions: int = 3, hidden: tuple[int, int] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.torso = nn.Sequential(*layers)
        self.logits = nn.Linear(last, n_actions)
        self.value_head = nn.Linear(last, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.logits.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.torso(x)
        return self.logits(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def sample(self, obs: np.ndarray) -> tuple[int, float, float]:
        logits, value = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    @torch.no_grad()
    def greedy(self, obs: np.ndarray) -> int:
        logits, _ = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())
