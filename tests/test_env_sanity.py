import dataclasses

import numpy as np

from baselines.agents import HeuristicAgent, RandomAgent
from environment.config import Config
from environment.pong_env import PongEnv


def test_reset_obs_shapes():
    env = PongEnv(seed=0)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"A1", "A2", "B1", "B2"}
    for o in obs.values():
        assert o.shape == (8,)
        assert np.all(np.isfinite(o))


def test_obs_ranges():
    env = PongEnv(seed=1)
    obs, _ = env.reset(seed=1)
    for _ in range(200):
        actions = {a: 0 for a in env.agent_ids}
        obs, _, term, trunc, _ = env.step(actions)
        for o in obs.values():
            assert -1.1 <= o[0] <= 1.1 and -1.1 <= o[3] <= 1.0 + 1e-6
        if all(term.values()) or all(trunc.values()):
            break


def test_reward_symmetry_on_score():
    env = PongEnv(seed=0)
    env.reset(seed=0)
    env.ball = [1.04, 0.0, 2.0, 0.0]  # about to exit right -> A scores
    _, rewards, _, _, _ = env.step({a: 0 for a in env.agent_ids})
    assert rewards["A1"] == rewards["A2"] == 1.0
    assert rewards["B1"] == rewards["B2"] == -1.0


def test_episode_ends_at_point_threshold():
    cfg = dataclasses.replace(Config(), points_to_win=1)
    env = PongEnv(config=cfg, seed=0)
    env.reset(seed=0)
    env.ball = [1.04, 0.0, 2.0, 0.0]
    _, _, terminated, _, _ = env.step({a: 0 for a in env.agent_ids})
    assert all(terminated.values())


def test_truncation_at_max_steps():
    cfg = dataclasses.replace(Config(), max_steps=5)
    env = PongEnv(config=cfg, seed=0)
    env.reset(seed=0)
    truncated = None
    for _ in range(6):
        _, _, _, truncated, _ = env.step({a: 0 for a in env.agent_ids})
    assert all(truncated.values())


def test_random_agents_run():
    env = PongEnv(seed=0)
    obs, _ = env.reset(seed=0)
    agents = {a: RandomAgent(seed=i) for i, a in enumerate(env.agent_ids)}
    for _ in range(50):
        actions = {a: ag.act(obs[a]) for a, ag in agents.items()}
        obs, _, term, trunc, _ = env.step(actions)
        if all(term.values()) or all(trunc.values()):
            break


def test_heuristic_agents_rally():
    env = PongEnv(seed=0)
    obs, _ = env.reset(seed=0)
    agents = {a: HeuristicAgent(team=a[0]) for a in env.agent_ids}
    rallied = False
    for _ in range(600):
        actions = {a: ag.act(obs[a]) for a, ag in agents.items()}
        obs, _, term, trunc, info = env.step(actions)
        if info["steps"] > 30:
            rallied = True
        if all(term.values()) or all(trunc.values()):
            break
    assert rallied  # heuristic paddles sustain at least a short rally


def test_determinism_same_seed():
    env1, env2 = PongEnv(seed=42), PongEnv(seed=42)
    o1, _ = env1.reset(seed=42)
    o2, _ = env2.reset(seed=42)
    for a in o1:
        assert np.allclose(o1[a], o2[a])


def test_serve_speed_varies_and_in_range():
    import math

    speeds = set()
    env = PongEnv(seed=0)
    for s in range(20):
        env.reset(seed=s)
        vx, vy = env.ball[2], env.ball[3]
        speed = math.hypot(vx, vy)
        assert env.cfg.serve_speed_min - 1e-6 <= speed <= env.cfg.serve_speed_max + 1e-6
        speeds.add(round(speed, 3))
    assert len(speeds) > 1  # serves differ across episodes


def test_2v2_spawn_formation_separated():
    env = PongEnv(seed=0)  # default 2v2
    env.reset(seed=0)
    assert env.paddles["A1"][0] > 0.2  # upper zone
    assert env.paddles["A2"][0] < -0.2  # lower zone
    assert env.paddles["B1"][0] > 0.2
    assert env.paddles["B2"][0] < -0.2


def test_shared_region_chooses_closest_paddle():
    env = PongEnv(seed=0)
    env.reset(seed=0)
    env.paddles["A1"] = [0.10, 0.0]
    env.paddles["A2"] = [0.00, 0.0]
    env.ball = [-0.90, 0.0, -1.0, 0.0]
    _, _, _, _, info = env.step({a: 0 for a in env.agent_ids})
    assert info["hits"]["A1"] == 0
    assert info["hits"]["A2"] == 1


def test_collision_selection_is_symmetric_for_team_b():
    env = PongEnv(seed=0)
    env.reset(seed=0)
    env.paddles["B1"] = [0.10, 0.0]
    env.paddles["B2"] = [0.00, 0.0]
    env.ball = [0.90, 0.0, 1.0, 0.0]
    _, _, _, _, info = env.step({a: 0 for a in env.agent_ids})
    assert info["hits"]["B1"] == 0
    assert info["hits"]["B2"] == 1


def test_calibrated_profile_is_frozen():
    cfg = Config()
    assert cfg.paddle_overlap == 0.15
    assert cfg.serve_speed_min == 0.805
    assert cfg.serve_speed_max == 1.15
    assert cfg.ball_speed_max == 1.84


def test_hits_give_no_reward():
    env = PongEnv(seed=0)
    env.reset(seed=0)
    env.paddles["A1"] = [0.0, 0.0]
    env.ball = [-0.90, 0.0, -1.0, 0.0]
    _, rewards, _, _, info = env.step({a: 0 for a in env.agent_ids})
    assert info["hits"]["A1"] == 1
    assert set(rewards.values()) == {0.0}  # point-only shared reward


def test_contact_regions_record_impact_band():
    env = PongEnv(seed=0)
    env.reset(seed=0)
    env.paddles["A2"] = [-0.6, 0.0]
    env.ball = [-0.90, -0.6, -1.0, 0.0]
    _, _, _, _, info = env.step({a: 0 for a in env.agent_ids})
    assert info["contact_regions"]["A2"] == {"upper": 1, "middle": 0, "lower": 0}
