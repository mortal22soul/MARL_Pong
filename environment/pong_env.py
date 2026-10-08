"""2v2 Pong environment with dict-based multi-agent API.

Agent IDs: A1, A2 (left) / B1, B2 (right). Slots: A1/B1 = upper (slot 0),
A2/B2 = lower (slot 1). No fixed region beyond the overlapping range limits.

Obs (8-dim, normalized): [own_y, own_vy, ball_x, ball_y, ball_vx, ball_vy,
mate_y, mate_vy]. Opponent positions excluded by design (TASK.md 3.4).
"""

import math
import random
from typing import ClassVar

import numpy as np
from gymnasium import spaces

from . import physics
from .config import DEFAULT, Config

STAY, UP, DOWN = 0, 1, 2


class PongEnv:
    metadata: ClassVar[dict] = {"render_modes": [None, "human"]}

    def __init__(self, config: Config = DEFAULT, render_mode=None, seed: int | None = None):
        self.cfg = config
        self.render_mode = render_mode
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        ids = list(config.agent_ids)
        self.agent_ids: list[str] = ids
        self.action_space = spaces.Dict({a: spaces.Discrete(3) for a in ids})
        lo = np.array([-1.0, -2.0, -1.1, -1.0, -2.0, -2.0, -1.0, -2.0], dtype=np.float32)
        hi = np.array([1.0, 2.0, 1.1, 1.0, 2.0, 2.0, 1.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Dict({a: spaces.Box(lo, hi, dtype=np.float32) for a in ids})
        self._screen = None
        self._clock = None
        self.reset(seed=seed)

    # --- state ---
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)
        # Paddles spawn in their own zones (upper/lower quartile of their range),
        # not stacked at center — like a doubles formation.
        self.paddles: dict[str, list[float]] = {}
        for a in self.agent_ids:
            lo, hi = self._allowed_range(a)
            y0 = lo + 0.75 * (hi - lo) if self._slot(a) == 0 else lo + 0.25 * (hi - lo)
            self.paddles[a] = [y0, 0.0]
        self.scores = {"A": 0, "B": 0}
        self.hits = {a: 0 for a in self.agent_ids}
        self.contact_regions = {a: {"upper": 0, "middle": 0, "lower": 0} for a in self.agent_ids}
        self.steps = 0
        self._serve()
        return {a: self._obs(a) for a in self.agent_ids}, {}

    def _slot(self, agent: str) -> int:
        return 0 if agent in ("A1", "B1") else 1

    def _allowed_range(self, agent: str) -> tuple[float, float]:
        """Movement limits for a paddle center."""
        return self.cfg.paddle_range(self._slot(agent))

    def _side(self, agent: str) -> str:
        return "left" if agent.startswith("A") else "right"

    @staticmethod
    def _contact_region(y: float) -> str:
        """Classify a return by the ball's vertical impact band."""
        if y < -1.0 / 3.0:
            return "upper"
        if y > 1.0 / 3.0:
            return "lower"
        return "middle"

    def _mate(self, agent: str) -> str:
        team = agent[0]
        return next(a for a in self.agent_ids if a.startswith(team) and a != agent)

    def _serve(self, toward: str | None = None):
        cfg = self.cfg
        ang = math.radians(self._rng.uniform(-cfg.serve_angle_deg_max, cfg.serve_angle_deg_max))
        direction = self._rng.choice([-1.0, 1.0]) if toward is None else toward
        speed = self._rng.uniform(cfg.serve_speed_min, cfg.serve_speed_max)
        vx = math.cos(ang) * speed * direction
        vy = math.sin(ang) * speed
        self.ball = [0.0, 0.0, vx, vy]

    # --- stepping ---
    def step(self, actions: dict[str, int]):
        cfg = self.cfg
        for a in self.agent_ids:
            act = int(actions.get(a, STAY))
            y, vy = self.paddles[a]
            y, vy = physics.step_paddle(y, vy, act, cfg)
            lo, hi = self._allowed_range(a)
            y = max(lo, min(hi, y))
            self.paddles[a] = [y, vy]
        px, py, vx, vy = self.ball
        px, py, vx, vy = physics.step_ball(px, py, vx, vy, cfg)
        # Pick from simultaneous candidates before velocity changes, so
        # overlapping teammates get no fixed-order priority. Closest vertical
        # center wins; agent ID breaks an exact, measure-zero tie consistently.
        side = "left" if vx < 0 else "right"
        candidates = [
            a
            for a in self.agent_ids
            if self._side(a) == side
            and physics.paddle_overlaps_ball(px, py, vx, self.paddles[a][0], side, cfg)
        ]
        if candidates:
            chosen = min(candidates, key=lambda a: (abs(py - self.paddles[a][0]), a))
            px, vx, vy, hit = physics.paddle_collision(
                px, py, vx, vy, self.paddles[chosen][0], side, cfg, self.paddles[chosen][1]
            )
            if hit:
                self.hits[chosen] += 1
                self.contact_regions[chosen][self._contact_region(py)] += 1
        self.ball = [px, py, vx, vy]
        self.steps += 1

        scorer = physics.check_score(px)
        rewards = {a: 0.0 for a in self.agent_ids}
        terminated = {a: False for a in self.agent_ids}
        if scorer is not None:
            self.scores[scorer] += 1
            for a in self.agent_ids:
                team = "A" if a.startswith("A") else "B"
                rewards[a] = 1.0 if team == scorer else -1.0
            if self.scores["A"] >= cfg.points_to_win or self.scores["B"] >= cfg.points_to_win:
                terminated = {a: True for a in self.agent_ids}
            else:
                self._serve()
        truncated = {a: self.steps >= cfg.max_steps for a in self.agent_ids}
        obs = {a: self._obs(a) for a in self.agent_ids}
        info = {
            "scores": dict(self.scores),
            "hits": dict(self.hits),
            "contact_regions": {a: dict(regions) for a, regions in self.contact_regions.items()},
            "steps": self.steps,
        }
        return obs, rewards, terminated, truncated, info

    def _obs(self, agent: str) -> np.ndarray:
        y, vy = self.paddles[agent]
        mate = self._mate(agent)
        my, mvy = self.paddles[mate]
        px, py, vx, vy = self.ball
        return np.array([y, vy, px, py, vx, vy, my, mvy], dtype=np.float32)

    # --- render ---
    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.cfg.screen_width, self.cfg.screen_height
        return int((x + 1.0) / 2.0 * w), int((y + 1.0) / 2.0 * h)

    def render(self):
        if self.render_mode != "human":
            return
        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.cfg.screen_width, self.cfg.screen_height))
            pygame.display.set_caption("marl-pong 2v2")
            self._clock = pygame.time.Clock()
        s = self._screen
        s.fill((10, 10, 18))
        # Center line + range guides.
        w, h = self.cfg.screen_width, self.cfg.screen_height
        import pygame as pg

        pg.draw.line(s, (60, 60, 80), (w // 2, 0), (w // 2, h), 2)
        ranges = [self.cfg.paddle_range(0), self.cfg.paddle_range(1)]
        for (lo, hi), color in zip(ranges, ((40, 90, 40), (90, 40, 40))):
            _, y0 = self._to_px(0, lo)
            _, y1 = self._to_px(0, hi)
            pg.draw.line(s, color, (0, y0), (w, y0), 1)
            pg.draw.line(s, color, (0, y1), (w, y1), 1)
        # Paddles.
        pw = max(4, int(self.cfg.paddle_width / 2.0 * w))
        ph = max(10, int(self.cfg.paddle_height / 2.0 * h))
        for a, (y, _) in self.paddles.items():
            side_left = a.startswith("A")
            x = -self.cfg.paddle_x_offset if side_left else self.cfg.paddle_x_offset
            cx, cy = self._to_px(x, y)
            color = (80, 160, 255) if side_left else (255, 120, 90)
            if a in ("A2", "B2"):
                color = tuple(c // 2 + 40 for c in color)
            pg.draw.rect(s, color, pg.Rect(cx - pw // 2, cy - ph // 2, pw, ph), border_radius=3)
        # Ball.
        bx, by = self._to_px(self.ball[0], self.ball[1])
        pg.draw.circle(s, (240, 240, 240), (bx, by), max(3, int(self.cfg.ball_radius / 2.0 * h)))
        # Score.
        font = pg.font.SysFont(None, 36)
        txt = font.render(f"{self.scores['A']} : {self.scores['B']}", True, (200, 200, 200))
        s.blit(txt, (w // 2 - txt.get_width() // 2, 10))
        pg.display.flip()
        self._clock.tick(self.cfg.fps)

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.quit()
            self._screen = None
            self._clock = None
