"""Greedy fixed policies loaded from PPO checkpoints for frozen-opponent runs."""

from __future__ import annotations

import numpy as np

from .policies import ActorCritic


class FrozenPolicy:
    def __init__(self, state_dict: dict, agent_id: str | None = None):
        self.net = ActorCritic(agent_id=agent_id)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def act(self, obs: np.ndarray) -> int:
        return self.net.greedy(obs)
