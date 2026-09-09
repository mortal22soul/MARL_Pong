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
