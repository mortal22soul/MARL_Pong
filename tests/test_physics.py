import math

from environment import physics
from environment.config import Config

CFG = Config()


def test_wall_bounce_top():
    px, py, vx, vy = physics.step_ball(0.0, -0.99, 0.0, -1.0, CFG)
    assert vy > 0
    assert py >= -1.0 + CFG.ball_radius - 1e-6


def test_wall_bounce_bottom():
    px, py, vx, vy = physics.step_ball(0.0, 0.99, 0.0, 1.0, CFG)
    assert vy < 0


def test_paddle_collision_left():
    y = 0.0
    px, vx, vy, hit = physics.paddle_collision(-CFG.paddle_x_offset, 0.0, -1.0, 0.0, y, "left", CFG)
    assert hit and vx > 0


def test_paddle_miss_no_hit():
    px, vx, vy, hit = physics.paddle_collision(-CFG.paddle_x_offset, 0.9, -1.0, 0.0, 0.0, "left", CFG)
    assert not hit


def test_score_detection():
    assert physics.check_score(1.2) == "A"
    assert physics.check_score(-1.2) == "B"
    assert physics.check_score(0.0) is None


def test_paddle_inertia():
    y, vy = 0.0, 0.0
    y2, vy2 = physics.step_paddle(y, vy, 2, CFG)
    assert vy2 > 0 and y2 > y  # accelerates down, no teleport
    assert abs(vy2) <= CFG.paddle_max_speed + 1e-6


def test_paddle_range_overlap():
    lo0, hi0 = CFG.paddle_range(0)
    lo1, hi1 = CFG.paddle_range(1)
    assert lo0 < hi1 and lo1 < hi0  # regions overlap
    assert hi0 > hi1 and lo1 < lo0  # but not identical
