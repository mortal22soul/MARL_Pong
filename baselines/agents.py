"""Scripted baselines: random + ball-tracking heuristic (no learning)."""

import random

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, obs: np.ndarray) -> int:
        return self._rng.choice([0, 1, 2])


class HeuristicAgent:
    """Move toward the ball's y when the ball approaches, else drift home.

    obs layout: [own_y, own_vy, ball_x, ball_y, ball_vx, ball_vy, mate_y, mate_vy].
    +y is DOWN, so ball above (smaller y) -> UP (1).
    home: where to hold when the ball is moving away (e.g. mid of own range,
    so teammates keep a doubles formation instead of piling at center).
    """

    def __init__(self, deadzone: float = 0.03, team: str = "A", home: float = 0.0):
        self.deadzone = deadzone
        self.team = team
        self.home = home

    def act(self, obs: np.ndarray) -> int:
        own_y = float(obs[0])
        ball_y = float(obs[3])
        ball_vx = float(obs[4])
        approaching = (self.team == "A" and ball_vx < 0) or (self.team == "B" and ball_vx > 0)
        target = ball_y if approaching else self.home
        if target < own_y - self.deadzone:
            return 1  # up
        if target > own_y + self.deadzone:
            return 2  # down
        return 0


class RangeAwareAgent:
    """Non-coordinating baseline that stays at its assigned range midpoint.

    It establishes how much positional separation comes from geometry and
    spawn/range constraints alone, rather than learned coordination.
    """

    def __init__(self, home: float, deadzone: float = 0.03):
        self.home = home
        self.deadzone = deadzone

    def act(self, obs: np.ndarray) -> int:
        own_y = float(obs[0])
        if self.home < own_y - self.deadzone:
            return 1
        if self.home > own_y + self.deadzone:
            return 2
        return 0


class PredictiveAgent(HeuristicAgent):
    """Same-information reference that targets the ball's paddle-line intercept.

    It uses only observed ball state plus fixed public arena geometry. This is
    a calibration reference, not a learning target or a privileged oracle.
    """

    def __init__(
        self,
        team: str,
        paddle_x: float,
        ball_radius: float,
        home: float = 0.0,
        deadzone: float = 0.03,
    ):
        super().__init__(deadzone=deadzone, team=team, home=home)
        self.paddle_x = paddle_x
        self.ball_radius = ball_radius

    def act(self, obs: np.ndarray) -> int:
        own_y, ball_x, ball_y, ball_vx, ball_vy = map(
            float, (obs[0], obs[2], obs[3], obs[4], obs[5])
        )
        approaching = (self.team == "A" and ball_vx < 0) or (self.team == "B" and ball_vx > 0)
        target = self.home
        if approaching and abs(ball_vx) > 1e-8:
            time_to_line = (self.paddle_x - ball_x) / ball_vx
            if time_to_line >= 0:
                target = _reflect_y(ball_y + ball_vy * time_to_line, self.ball_radius)
        if target < own_y - self.deadzone:
            return 1
        if target > own_y + self.deadzone:
            return 2
        return 0


def _reflect_y(y: float, radius: float) -> float:
    """Map an unconstrained y through repeated top/bottom reflections."""
    limit = 1.0 - radius
    period = 4.0 * limit
    folded = (y + limit) % period
    if folded <= 2.0 * limit:
        return folded - limit
    return 3.0 * limit - folded
