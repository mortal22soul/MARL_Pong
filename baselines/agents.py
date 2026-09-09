"""Scripted baselines: random + ball-tracking heuristic (no learning)."""

import random

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, obs: np.ndarray) -> int:
        return self._rng.choice([0, 1, 2])


class HeuristicAgent:
    """Move toward the ball's y when the ball approaches, else drift to range center.

    obs layout: [own_y, own_vy, ball_x, ball_y, ball_vx, ball_vy, mate_y, mate_vy].
    +y is DOWN, so ball above (smaller y) -> UP (1).
    """

    def __init__(self, deadzone: float = 0.03, team: str = "A"):
        self.deadzone = deadzone
        self.team = team

    def act(self, obs: np.ndarray) -> int:
        own_y, _, ball_x, ball_y, ball_vx = (
            float(obs[0]),
            float(obs[1]),
            float(obs[2]),
            float(obs[3]),
            float(obs[4]),
        )
        approaching = (self.team == "A" and ball_vx < 0) or (self.team == "B" and ball_vx > 0)
        target = ball_y if approaching else 0.0
        if target < own_y - self.deadzone:
            return 1  # up
        if target > own_y + self.deadzone:
            return 2  # down
        return 0


def make_heuristic_team(team: str, **kw) -> HeuristicAgent:
    return HeuristicAgent(team=team, **kw)
