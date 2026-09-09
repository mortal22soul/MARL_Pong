"""Pure physics helpers — no pygame, no numpy required (plain floats)."""

from .config import Config


def step_paddle(y: float, vy: float, action: int, cfg: Config) -> tuple[float, float]:
    """Integrate one paddle. action: 0=stay, 1=up, 2=down. +y is DOWN (screen-like)."""
    target = 0.0
    if action == 1:
        target = -cfg.paddle_max_speed
    elif action == 2:
        target = cfg.paddle_max_speed
    # Exponential approach gives minimal inertia (no teleport, no long slide).
    blend = min(1.0, cfg.paddle_accel * cfg.dt)
    vy_new = vy + (target - vy) * blend
    return y + vy_new * cfg.dt, vy_new


def clamp_paddle(y: float, cfg: Config, slot: int) -> float:
    lo, hi = cfg.paddle_range(slot)
    return max(lo, min(hi, y))


def step_ball(
    px: float, py: float, vx: float, vy: float, cfg: Config
) -> tuple[float, float, float, float]:
    px += vx * cfg.dt
    py += vy * cfg.dt
    # Top/bottom walls at y = +/-1.
    if py - cfg.ball_radius < -1.0:
        py = -1.0 + cfg.ball_radius
        vy = abs(vy)
    elif py + cfg.ball_radius > 1.0:
        py = 1.0 - cfg.ball_radius
        vy = -abs(vy)
    return px, py, vx, vy


def paddle_collision(
    px: float, py: float, vx: float, vy: float, paddle_y: float, side: str, cfg: Config
) -> tuple[float, float, float, bool]:
    """Reflect off a paddle if overlapping. Returns (px, vx, vy, hit).

    side: 'left' (Team A, ball moving -x) or 'right' (Team B, ball moving +x).
    Adds a small vy kick based on hit offset + clamps speed.
    """
    h = cfg.paddle_height / 2.0
    w = cfg.paddle_width / 2.0 + cfg.ball_radius
    x_line = -cfg.paddle_x_offset if side == "left" else cfg.paddle_x_offset
    hit = False
    if side == "left" and vx < 0 and abs(px - x_line) <= w and abs(py - paddle_y) <= h:
        px = x_line + w
        vx = abs(vx) * 1.02
        vy += (py - paddle_y) * 1.5
        hit = True
    elif side == "right" and vx > 0 and abs(px - x_line) <= w and abs(py - paddle_y) <= h:
        px = x_line - w
        vx = -abs(vx) * 1.02
        vy += (py - paddle_y) * 1.5
        hit = True
    if hit:
        speed = (vx * vx + vy * vy) ** 0.5
        if speed > cfg.ball_speed_max:
            scale = cfg.ball_speed_max / speed
            vx *= scale
            vy *= scale
    return px, vx, vy, hit


def check_score(px: float) -> str | None:
    """'A' if Team A scores (ball exits right), 'B' if Team B scores, else None."""
    if px > 1.05:
        return "A"
    if px < -1.05:
        return "B"
    return None
