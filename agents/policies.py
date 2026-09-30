"""Actor-critic MLPs for independent PPO with canonical feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def extract_features(obs: np.ndarray, agent: str, paddle_x_offset: float = 0.92) -> np.ndarray:
    """Transform raw physical observation into canonical relative features.

    Raw obs layout: [own_y, own_vy, px, py, vx, vy, my, mvy].
    Canonical layout: [own_y, own_vy, dx_ball, dy_ball, approach_vx, vy, dy_mate, mvy].
    - dx_ball: distance to ball in front of paddle (always >= 0 when ball in arena).
    - dy_ball: vertical displacement (ball_y - own_y); positive means ball is below paddle.
    - approach_vx: positive when ball moves toward own paddle, negative when moving away.
    - dy_mate: teammate vertical displacement (mate_y - own_y).
    """
    own_y, own_vy, px, py, vx, vy, my, mvy = (float(v) for v in obs)
    is_left = agent.startswith("A")
    paddle_x = -paddle_x_offset if is_left else paddle_x_offset
    dx_ball = (px - paddle_x) if is_left else (paddle_x - px)
    approach_vx = -vx if is_left else vx
    dy_ball = py - own_y
    dy_mate = my - own_y
    return np.array(
        [own_y, own_vy, dx_ball, dy_ball, approach_vx, vy, dy_mate, mvy],
        dtype=np.float32,
    )


class ActorCritic(nn.Module):
    """Actor-Critic MLP with decoupled actor and critic representations."""

    def __init__(
        self,
        obs_dim: int = 8,
        n_actions: int = 3,
        hidden: tuple[int, int] = (64, 64),
        agent_id: str | None = None,
    ):
        super().__init__()
        self.agent_id = agent_id

        # Actor torso & head
        actor_layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            actor_layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.actor_torso = nn.Sequential(*actor_layers)
        self.logits = nn.Linear(last, n_actions)

        # Critic torso & head
        critic_layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            critic_layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.critic_torso = nn.Sequential(*critic_layers)
        self.value_head = nn.Linear(last, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.logits.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @classmethod
    def _load_state_dict_compat(cls, state_dict: dict) -> dict:
        """Load state dict from either new (actor_torso/critic_torso) or old (torso) format."""
        # Check if it's the old format (single shared torso)
        if "torso.0.weight" in state_dict:
            # Convert old format to new format
            new_state = {}
            # Copy torso weights to both actor and critic torsos
            for key, value in state_dict.items():
                if key.startswith("torso."):
                    layer_name = key[6:]  # Remove "torso."
                    new_state[f"actor_torso.{layer_name}"] = value.clone()
                    new_state[f"critic_torso.{layer_name}"] = value.clone()
                elif key.startswith(("logits.", "value_head.")):
                    new_state[key] = value.clone()
                else:
                    new_state[key] = value.clone()
            return new_state
        return state_dict

    @staticmethod
    def is_legacy_state_dict(state_dict: dict) -> bool:
        """Whether weights predate separate torsos and canonical features."""
        return "torso.0.weight" in state_dict

    def load_checkpoint(self, state_dict: dict) -> bool:
        """Load current or legacy weights without changing their input semantics.

        Legacy checkpoints used one raw-observation torso. Their copied weights
        remain valid only with raw observations, so feature canonicalization is
        disabled for that network after conversion.
        """
        legacy = self.is_legacy_state_dict(state_dict)
        self.load_state_dict(self._load_state_dict_compat(state_dict))
        if legacy:
            self.agent_id = None
        return legacy

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logits(self.actor_torso(x)), self.value_head(self.critic_torso(x)).squeeze(-1)

    def _prep_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.agent_id is not None:
            return extract_features(obs, self.agent_id)
        return obs

    @torch.no_grad()
    def sample(self, obs: np.ndarray) -> tuple[int, float, float]:
        feat = self._prep_obs(obs)
        logits, value = self(torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    @torch.no_grad()
    def greedy(self, obs: np.ndarray) -> int:
        feat = self._prep_obs(obs)
        logits, _ = self(torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())
