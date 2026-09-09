"""Single source of truth for 2v2 Pong tuning.

Internal sim logic runs in normalized coordinates [-1, 1] on both axes.
Rendering maps those to pixels; physics never touches pixels.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # --- simulation ---
    dt: float = 1.0 / 60.0
    # Ball serve: uniform angle in [-max, +max] degrees off horizontal, random side,
    # and uniform speed in [serve_speed_min, serve_speed_max] so serves vary.
    serve_speed_min: float = 0.7
    serve_speed_max: float = 1.0
    ball_speed_max: float = 1.6  # clamp after rally escalation
    serve_angle_deg_max: float = 35.0
    ball_radius: float = 0.025
    # Rally escalation: vx multiplied by hit_speedup on every paddle contact,
    # plus a "smash" kick transferring paddle motion into the ball.
    hit_speedup: float = 1.03
    paddle_smash_factor: float = 0.35

    # --- paddles ---
    paddle_height: float = 0.34
    paddle_width: float = 0.03
    paddle_max_speed: float = 1.6  # normalized units / second
    paddle_accel: float = 12.0  # approach to target velocity (inertia)
    # Vertical range each paddle may occupy, as (center, half_height) in y.
    # Overlap is the TEMPORARY default; final value is frozen later via
    # heuristic-only sweep per TASK.md (not tuned to PPO).
    # P1 covers upper region, P2 covers lower region, overlapping in middle.
    paddle_overlap: float = 0.4
    paddle_x_offset: float = 0.92  # |x| of paddle centerlines

    # --- episode ---
    points_to_win: int = 5
    max_steps: int = 2000  # prevents infinite rallies stalling training
    # "2v2" (A1,A2 vs B1,B2, overlapping partial ranges) or
    # "1v1" (A1 vs B1, each covering the full field height).
    mode: str = "2v2"

    # --- render ---
    screen_width: int = 800
    screen_height: int = 600
    fps: int = 60

    agent_ids: tuple = field(default=("A1", "A2", "B1", "B2"))

    def paddle_range(self, slot: int) -> tuple[float, float]:
        """Return (y_min, y_max) center limits for paddle slot 0 (upper) or 1 (lower).

        Full field half-height is 1.0; paddle center must stay within
        [-1 + h/2, 1 - h/2]. The overlap parameter blends between disjoint
        halves (overlap=0) and full-field roaming (overlap=1).
        """
        limit = 1.0 - self.paddle_height / 2.0
        mid = 0.0
        span = limit  # max excursion from mid for full coverage
        # Upper paddle: [mid - span*overlap ... limit]; lower mirrored.
        if slot == 0:
            lo = mid - span * self.paddle_overlap
            hi = limit
        else:
            lo = -limit
            hi = mid + span * self.paddle_overlap
        return (lo, hi)


DEFAULT = Config()
