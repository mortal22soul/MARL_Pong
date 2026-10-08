"""Tests for scripted baselines: random policy bounds, heuristic targeting."""

import numpy as np

from baselines.agents import HeuristicAgent, RandomAgent
from environment.config import Config
from environment.pong_env import PongEnv


def test_random_agent_stays_in_action_space():
    agent = RandomAgent(seed=7)
    obs = np.zeros(8)
    assert all(agent.act(obs) in (0, 1, 2) for _ in range(100))


def test_random_agent_seeded_is_deterministic():
    obs = np.zeros(8)
    seq1 = [RandomAgent(seed=3).act(obs) for _ in range(50)]
    seq2 = [RandomAgent(seed=3).act(obs) for _ in range(50)]
    assert seq1 == seq2


def _obs(own_y=0.0, ball_y=0.0, ball_vx=0.0):
    return np.array([own_y, 0.0, 0.5, ball_y, ball_vx, 0.0, 0.0, 0.0])


def test_heuristic_moves_toward_approaching_ball():
    agent = HeuristicAgent(team="A")  # +y is down: ball above -> smaller y -> UP
    assert agent.act(_obs(own_y=0.0, ball_y=-0.5, ball_vx=-0.8)) == 1
    assert agent.act(_obs(own_y=0.0, ball_y=0.5, ball_vx=-0.8)) == 2


def test_heuristic_holds_deadzone():
    agent = HeuristicAgent(team="A", deadzone=0.05)
    assert agent.act(_obs(own_y=0.0, ball_y=0.02, ball_vx=-0.8)) == 0


def test_heuristic_drifts_home_when_ball_recedes():
    agent = HeuristicAgent(team="A", home=0.25)
    # Ball moving away (vx > 0 for team A): target is home, below own y -> down.
    assert agent.act(_obs(own_y=0.0, ball_y=-0.5, ball_vx=0.8)) == 2
    agent_home_above = HeuristicAgent(team="A", home=-0.25)
    assert agent_home_above.act(_obs(own_y=0.0, ball_y=-0.5, ball_vx=0.8)) == 1


def test_heuristic_team_vs_random_team_plays_full_episodes():
    import dataclasses

    cfg = dataclasses.replace(Config(), max_steps=400, points_to_win=1)
    env = PongEnv(config=cfg, seed=0)
    heuristics = {}
    for a in env.agent_ids:
        lo, hi = env._allowed_range(a)
        heuristics[a] = HeuristicAgent(team=a[0], home=(lo + hi) / 2.0)
    randoms = {a: RandomAgent(seed=100 + i) for i, a in enumerate(env.agent_ids)}
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        actions = {
            a: heuristics[a].act(obs[a]) if a.startswith("A") else randoms[a].act(obs[a])
            for a in env.agent_ids
        }
        obs, _, term, trunc, info = env.step(actions)
        done = all(term.values()) or all(trunc.values())
    # A coordinated tracking team should beat a random team at least once.
    assert info["scores"]["A"] >= 1
    assert info["steps"] <= 400
